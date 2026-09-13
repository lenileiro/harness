"""Inspectable setup and offline-first environment diagnostics."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Literal

import tomlkit
import typer
from tomlkit.items import InlineTable, Table

from harness.adapters.claude import claude_cli_available, inspect_claude_cli_auth
from harness.adapters.codex import codex_cli_available, inspect_codex_cli_auth
from harness.adapters.openai import inspect_codex_openai_auth
from harness.cli.common import KNOWN_PROVIDERS
from harness.cli.config import HarnessConfig, default_config_path, load_config
from harness.core.skills import SkillLibrary, default_skill_paths
from harness.storage.sqlite import default_db_path
from harness.tools.mcp import MCPServerConfig, MCPToolset


def setup_command(
    provider: Annotated[str, typer.Option("--provider", help="Provider to use by default.")],
    model: Annotated[
        str, typer.Option("--model", help="Model identifier for the selected provider.")
    ],
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Update an existing config, preserving unrelated settings."),
    ] = False,
) -> None:
    """Write provider/model defaults without storing credentials or logging in."""
    if not model.strip():
        raise typer.BadParameter("model must be nonempty")
    target = (config_path or default_config_path()).expanduser().absolute()
    temporary: Path | None = None
    try:
        if target.is_symlink():
            raise ValueError("config is a symlink; specify its target explicitly")
        if target.exists() and not force:
            raise ValueError("config already exists; use --force to update provider/model defaults")
        document = tomlkit.parse(target.read_text()) if target.exists() else tomlkit.document()
        if (
            provider not in KNOWN_PROVIDERS
            and load_config(target).provider(provider).get("driver") != "openai-compatible"
        ):
            raise ValueError(
                "Choose a built-in provider or configure [provider.NAME] with driver='openai-compatible' first"
            )
        defaults = document.get("default")
        if defaults is None:
            defaults = tomlkit.table()
            document["default"] = defaults
        if not isinstance(defaults, Table | InlineTable):
            raise ValueError("[default] must be a table")
        defaults["provider"] = provider
        defaults["model"] = model.strip()
        content = tomlkit.dumps(document)
        target.parent.mkdir(parents=True, exist_ok=True)
        # mkstemp is private by default; validate the complete replacement before
        # atomically replacing anything, including when --force is supplied.
        fd, name = tempfile.mkstemp(prefix=".harness-config-", suffix=".toml", dir=target.parent)
        temporary = Path(name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        load_config(temporary)
        if force:
            temporary.replace(target)
        else:
            # An exclusive hard link prevents a concurrent setup from silently
            # overwriting a config created after the initial existence check.
            os.link(temporary, target)
            temporary.unlink()
        temporary = None
    except (ValueError, OSError) as exc:
        # Parser exceptions may embed source text; never echo config contents.
        if isinstance(exc, ValueError) and not type(exc).__module__.startswith(
            ("tomlkit", "harness.cli.config")
        ):
            detail = str(exc)
        else:
            detail = f"{type(exc).__name__}; inspect the config syntax and filesystem permissions"
        typer.echo(f"Setup failed: {detail}", err=True)
        raise typer.Exit(1) from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    typer.echo(f"Saved configuration to {target}. Run harness doctor to check the environment.")


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: Literal["ok", "warning", "error"]
    detail: str


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _probe_directory(path: Path) -> None:
    """Exercise actual write access without creating persistent workspace state."""
    if not path.is_dir():
        raise OSError("not a directory")
    with tempfile.TemporaryFile(dir=path):
        pass


def _probe_database(path: Path) -> None:
    if path.exists():
        if not path.is_file():
            raise OSError("database path is not a file")
        with path.open("r+b"):
            pass
    directory = path.parent
    while not directory.exists():
        directory = directory.parent
    _probe_directory(directory)


def _credentials(provider: str) -> tuple[bool, str]:
    if provider == "ollama":
        return True, "no API key required; endpoint has not been probed"
    if provider == "codex":
        auth = inspect_codex_cli_auth() or {}
        present = bool(auth.get("has_openai_api_key") or auth.get("has_access_token"))
        if not codex_cli_available():
            return False, "Codex CLI executable is missing"
        return (
            present,
            "saved login present; not verified" if present else "saved Codex login is missing",
        )
    if provider == "claude":
        if not claude_cli_available():
            return False, "Claude Code CLI executable is missing"
        auth = inspect_claude_cli_auth() or {}
        if not auth.get("logged_in"):
            return False, "Claude Code login is missing; run `claude auth login`"
        return True, f"{auth.get('auth_method', 'unknown')} login present; not verified"
    variable = {
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }[provider]
    present = bool(os.environ.get(variable, "").strip())
    if provider == "openai":
        present = present or bool((inspect_codex_openai_auth() or {}).get("has_openai_api_key"))
    return present, "credential present; not verified" if present else f"{variable} is missing"


def _mcp_executable_available(server: MCPServerConfig, cwd: Path, env: dict[str, str]) -> bool:
    command = server.command or ""
    if os.path.dirname(command):
        directory = server.cwd or cwd
        if not directory.is_absolute():
            directory = cwd / directory
        target = directory / command
        return target.is_file() and os.access(target, os.X_OK)
    return shutil.which(command, path=env.get("PATH", os.environ.get("PATH"))) is not None


async def _check_mcp(servers: list[MCPServerConfig], cwd: Path) -> list[DoctorCheck]:
    checks = []
    for server in servers:
        try:
            async with MCPToolset([server], cwd=cwd) as toolset:
                checks.append(
                    DoctorCheck(
                        f"mcp:{server.name}:connection",
                        "ok",
                        f"connected; {len(toolset.tools)} tools discovered",
                    )
                )
        except Exception as exc:
            checks.append(
                DoctorCheck(
                    f"mcp:{server.name}:connection",
                    "error",
                    f"connection failed ({type(exc).__name__}); check server and authentication",
                )
            )
    return checks


def collect_doctor_checks(
    *,
    config_path: Path | None,
    cwd: Path,
    db: Path | None = None,
    connect: bool = False,
) -> list[DoctorCheck]:
    """No provider calls, plugin imports, server launches or logins by default."""
    checks: list[DoctorCheck] = []
    target = config_path or default_config_path()
    config = HarnessConfig()
    try:
        if config_path is not None and not target.is_file():
            raise ValueError("explicit configuration file is missing")
        config = load_config(target)
        checks.append(
            DoctorCheck(
                "config",
                "ok" if target.exists() else "warning",
                "configuration parsed"
                if target.exists()
                else "optional config missing; built-in defaults apply",
            )
        )
    except (ValueError, OSError) as exc:
        checks.append(
            DoctorCheck(
                "config",
                "error",
                f"configuration could not be read or validated ({type(exc).__name__})",
            )
        )

    selected = config.default_provider or "ollama"
    custom = {
        name: settings
        for name, settings in config.provider_settings.items()
        if settings.get("driver") == "openai-compatible"
    }
    if selected not in KNOWN_PROVIDERS and selected not in custom:
        checks.append(
            DoctorCheck("provider:default", "error", "configured default provider is unsupported")
        )
    for provider in KNOWN_PROVIDERS:
        present, detail = _credentials(provider)
        status: Literal["ok", "warning", "error"] = (
            "ok" if present else ("error" if provider == selected else "warning")
        )
        if provider == "ollama":
            status = "warning"
        checks.append(DoctorCheck(f"provider:{provider}", status, detail))
    for name, settings in custom.items():
        variable = settings.get("api_key_env")
        configured = (
            isinstance(settings.get("base_url"), str)
            and isinstance(variable, str)
            and (not variable or bool(os.environ.get(variable)))
        )
        if settings.get("oauth") is not None:
            try:
                from harness.cli.account_auth import AccountAuth, OAuthAccountConfig

                state = AccountAuth(
                    name,
                    OAuthAccountConfig.model_validate(settings["oauth"]),
                    resource=settings["base_url"],
                ).status()
                configured = state["configured"] and not state["quarantined"]
            except (ValueError, KeyError, OSError):
                configured = False
        checks.append(
            DoctorCheck(
                f"provider:{name}",
                "ok" if configured else "error",
                "explicit endpoint and credentials configured; not probed"
                if configured
                else "requires base_url and api_key_env (empty string permits an unauthenticated endpoint)",
            )
        )
    if config.execution is not None:
        execution = config.execution
        command = {
            "local": execution.shell,
            "docker": execution.docker_binary,
            "ssh": execution.ssh_binary,
            "singularity": execution.singularity_binary,
        }.get(execution.backend)
        module = {"modal": "modal", "daytona": "daytona", "vercel_sandbox": "vercel.sandbox"}.get(
            execution.backend
        )
        available = (
            bool(shutil.which(command)) if command else bool(module and _module_available(module))
        )
        checks.append(
            DoctorCheck(
                f"execution:{execution.backend}",
                "ok" if available else "error",
                "configured backend dependency available; execution/authentication not probed"
                if available
                else "configured backend dependency missing; install its optional extra or executable",
            )
        )
    if config.browser is not None:
        available = _module_available("playwright")
        checks.append(
            DoctorCheck(
                f"browser:{config.browser.backend}",
                "ok" if available else "error",
                "Playwright available; browser binary and connection require a runtime check"
                if available
                else "install the browser package and Playwright browser runtime",
            )
        )
    if config.media.enabled:
        remote = any(
            (config.media.image_model, config.media.transcription_model, config.media.speech_model)
        )
        available = not remote or bool(os.environ.get(config.media.api_key_env))
        checks.append(
            DoctorCheck(
                "media",
                "ok" if available else "error",
                "media tools configured; external services not probed"
                if available
                else "configured media credential is missing",
            )
        )
    if config.computer.enabled:
        available = _module_available("pyautogui") and _module_available("PIL")
        checks.append(
            DoctorCheck(
                "computer",
                "ok" if available else "error",
                "desktop dependencies available; graphical session and OS permissions not probed"
                if available
                else "install tools-computer[desktop]; no desktop action was attempted",
            )
        )
    for name, enabled, variable in (
        ("honcho", config.honcho.enabled, config.honcho.api_key_env),
        ("homeassistant", config.homeassistant.enabled, config.homeassistant.token_env),
    ):
        if enabled:
            present = bool(os.environ.get(variable))
            checks.append(
                DoctorCheck(
                    name,
                    "ok" if present else "error",
                    "credential reference configured; service not contacted"
                    if present
                    else "configured credential environment reference is missing",
                )
            )
    if config.a2a.enabled:
        if not config.a2a.peers:
            checks.append(DoctorCheck("a2a", "warning", "enabled with no remote peers configured"))
        for name, peer in config.a2a.peers.items():
            present = not peer.token_env or bool(os.environ.get(peer.token_env))
            checks.append(
                DoctorCheck(
                    f"a2a:{name}",
                    "ok" if present else "error",
                    "endpoint and credential reference configured; peer not contacted"
                    if present
                    else "configured peer credential environment reference is missing",
                )
            )
    if config.portal.enabled:
        try:
            from harness.cli.account_auth import AccountAuth, OAuthAccountConfig

            settings = config.provider(config.portal.provider)
            account_state = AccountAuth(
                config.portal.provider,
                OAuthAccountConfig.model_validate(settings.get("oauth")),
                resource=settings.get("base_url", ""),
            ).status()
            present = account_state["configured"] and not account_state["quarantined"]
        except (ValueError, KeyError, OSError):
            present = False
        checks.append(
            DoctorCheck(
                "portal",
                "ok" if present else "error",
                "account state present; routes and authorization not probed"
                if present
                else "configure and authorize the selected portal provider account",
            )
        )
    for name in ("aiosqlite", "httpx", "mcp", "yaml", "tomlkit"):
        checks.append(
            DoctorCheck(
                f"dependency:{name}",
                "ok" if _module_available(name) else "error",
                "module available"
                if _module_available(name)
                else "module is missing; reinstall Harness dependencies",
            )
        )
    if selected == "anthropic":
        checks.append(
            DoctorCheck(
                "dependency:anthropic",
                "ok" if _module_available("anthropic") else "error",
                "SDK available" if _module_available("anthropic") else "Anthropic SDK is missing",
            )
        )
    for command in ("git", "sh"):
        checks.append(
            DoctorCheck(
                f"executable:{command}",
                "ok" if shutil.which(command) else "warning",
                "executable available" if shutil.which(command) else "executable is missing",
            )
        )

    try:
        _probe_directory(cwd)
        checks.append(DoctorCheck("workspace", "ok", "directory is writable"))
    except OSError:
        checks.append(
            DoctorCheck("workspace", "error", "workspace directory is missing or not writable")
        )
    workspace_db = cwd / ".harness" / "harness.db"
    database = db or (workspace_db if workspace_db.exists() else default_db_path())
    try:
        _probe_database(database)
        checks.append(
            DoctorCheck(
                "database",
                "ok",
                "database file/directory is writable"
                if database.exists()
                else "database directory can be created; database has not been initialized",
            )
        )
    except OSError:
        checks.append(
            DoctorCheck("database", "error", "database path or its parent is not writable")
        )

    if config.skills_enabled:
        roots = [(cwd / Path(path).expanduser()).resolve() for path in config.skill_paths]
        library = SkillLibrary.load(roots + default_skill_paths(cwd))
        checks.append(
            DoctorCheck(
                "skills",
                "error" if library.errors else "ok",
                f"{len(library.skills)} valid skills; {len(library.errors)} invalid; {len(library.shadowed)} shadowed",
            )
        )
        for path in library.errors:
            checks.append(
                DoctorCheck(
                    f"skill:{path}",
                    "error",
                    "skill validation failed; use harness skills validate to inspect",
                )
            )
    else:
        checks.append(DoctorCheck("skills", "ok", "skill loading disabled"))

    available_servers = []
    for server in config.mcp_servers:
        if not server.enabled:
            checks.append(DoctorCheck(f"mcp:{server.name}", "ok", "disabled; not connected"))
            continue
        try:
            env = server.resolve_env()
            server.resolve_headers()
            if server.transport == "stdio" and not _mcp_executable_available(server, cwd, env):
                raise ValueError("configured server executable is missing")
        except ValueError:
            checks.append(
                DoctorCheck(
                    f"mcp:{server.name}",
                    "error",
                    "required environment reference or server executable is missing",
                )
            )
            continue
        checks.append(
            DoctorCheck(
                f"mcp:{server.name}",
                "ok",
                "configuration and credential references valid; not connected",
            )
        )
        available_servers.append(server)
    if (
        selected == "codex"
        and available_servers
        and config.provider("codex").get("mode", "exec") != "app-server"
    ):
        checks.append(
            DoctorCheck(
                "mcp:provider",
                "error",
                "Codex CLI cannot consume Harness MCP tools; choose a native tool-capable provider",
            )
        )
    if connect:
        checks.extend(asyncio.run(_check_mcp(available_servers, cwd)))
    return checks


def doctor_command(
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
    cwd: Annotated[Path | None, typer.Option("--cwd")] = None,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    connect: Annotated[
        bool, typer.Option("--connect", help="Actually connect enabled MCP servers for discovery.")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Check configuration, credentials and local dependencies; no model calls."""
    checks = collect_doctor_checks(
        config_path=config_path, cwd=(cwd or Path.cwd()).resolve(), db=db, connect=connect
    )
    ok = all(check.status != "error" for check in checks)
    if json_output:
        typer.echo(json.dumps({"ok": ok, "checks": [asdict(check) for check in checks]}))
    else:
        for check in checks:
            typer.echo(f"{check.status.upper():7} {check.name}: {check.detail}")
    if not ok:
        raise typer.Exit(1)
