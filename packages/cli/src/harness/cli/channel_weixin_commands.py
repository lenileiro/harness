"""Explicit QR onboarding for a selected workspace's Weixin channel."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import typer

from harness.cli.channels.transports import ChannelError
from harness.cli.channels.weixin_pairing import WeixinPairing
from harness.core.paths import read_regular_file


def credential_path(cwd: Path) -> Path:
    root = cwd.resolve()
    current = root
    for component in (".harness", "channels", "weixin-account.json"):
        current /= component
        if current.is_symlink() or (
            current.exists() and getattr(current.lstat(), "st_file_attributes", 0) & 0x400
        ):
            raise ValueError("Weixin credential path must not contain symlinks or reparse points")
    if not current.resolve().is_relative_to(root):
        raise ValueError("Weixin credential path escapes the workspace")
    return current


def save_credentials(cwd: Path, account: dict[str, str], *, replace: bool = False) -> Path:
    target = credential_path(cwd)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.exists() and not replace:
        raise ValueError("Weixin credentials exist; use --replace to pair a replacement explicitly")
    if target.exists():
        read_regular_file(target, max_bytes=16384)
    descriptor, temporary = tempfile.mkstemp(prefix=".weixin-", dir=target.parent)
    staged = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.chmod(staged, 0o600)
            json.dump(account, stream)
            stream.flush()
            os.fsync(stream.fileno())
        credential_path(cwd)
        if replace:
            os.replace(staged, target)
        else:
            # Atomic no-clobber install, including a concurrent pairing process.
            os.link(staged, target)
        return target
    finally:
        staged.unlink(missing_ok=True)


def register_weixin_commands(app: typer.Typer) -> None:
    @app.command("weixin-pair")
    def weixin_pair(
        cwd: Path | None = typer.Option(None, "--cwd"),
        replace: bool = typer.Option(False, "--replace"),
        timeout: int = typer.Option(300, "--timeout", min=1, max=600),
    ) -> None:
        """Pair through Tencent's QR flow; save private credentials without printing tokens."""
        working_dir = (cwd or Path.cwd()).resolve()
        if not working_dir.is_dir():
            raise typer.BadParameter("--cwd must be an existing directory")
        try:
            target = credential_path(working_dir)
            if target.exists() and not replace:
                raise ValueError(
                    "Weixin credentials exist; use --replace to pair a replacement explicitly"
                )

            async def run():
                async def display(value: str):
                    import qrcode

                    qr = qrcode.QRCode(border=1)
                    qr.add_data(value)
                    qr.print_ascii(invert=True)
                    typer.echo("Scan this QR code with the Weixin account to connect.")

                async def code() -> str:
                    from prompt_toolkit import PromptSession

                    return await PromptSession[str]().prompt_async(
                        "Weixin verification code: ", is_password=True
                    )

                async with WeixinPairing() as pairing:
                    return await pairing.pair(
                        display_qr=display, verification_code=code, timeout_seconds=timeout
                    )

            account = asyncio.run(run())
            target = save_credentials(working_dir, account, replace=replace)
        except KeyboardInterrupt:
            return
        except (ValueError, ChannelError, TimeoutError) as exc:
            typer.echo(str(exc) or "Weixin pairing timed out", err=True)
            raise typer.Exit(1) from None
        except Exception:
            typer.echo("Weixin pairing failed; account credentials were not printed", err=True)
            raise typer.Exit(1) from None
        typer.echo("Credentials saved. Add to the selected profile configuration:")
        typer.echo("[channels.weixin]")
        typer.echo("account_file = " + json.dumps(str(target)))
        typer.echo("allowed_users = " + json.dumps([account["ilink_user_id"]]))
