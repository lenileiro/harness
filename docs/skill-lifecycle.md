# Portable skill lifecycle and learning

`harness skills` manages portable `SKILL.md` packages. Installation copies data;
it never runs repository hooks, package installers, or bundled scripts. Scripts
remain resources that ordinary tools may run under their normal approval policy.
Skill instructions and `allowed-tools` metadata never grant permissions.

## Install, inspect, update, and recover

```sh
harness skills install ./my-skills/check-output
harness skills install-git check-output https://github.com/OWNER/REPOSITORY \
  --commit FULL_40_CHARACTER_LOWERCASE_COMMIT_ID --path skills/check-output
harness skills inspect check-output
```

Remote installation supports public HTTPS GitHub and GitLab repositories. A full
commit ID is required; branch names, tags, credentials embedded in URLs, and
arbitrary archive hosts are rejected. Downloads are limited to 32 MiB compressed
and 64 MiB expanded. A selected package may contain at most 16 MiB and 1,000 files.
Archive links, special files, duplicate paths, and traversal paths are rejected.
Package frontmatter must validate and its declared name must match the requested
installation name. Private repositories and self-hosted Git servers currently
require an operator-prepared local package.

GitHub documents commit IDs as the reproducible reference for source archives;
GitLab's archive API accepts the pinned `sha`. Harness retains the repository,
commit, selected path, archive hash, and installed content digest. These are
source provenance, not a publisher signature or a security endorsement.
See [GitHub archives](https://docs.github.com/en/repositories/working-with-files/using-files/downloading-source-code-archives)
and [GitLab repository archives](https://docs.gitlab.com/api/repositories/#get-file-archive).

Copy the current `revision` from `skills inspect` into `--expected`:

```sh
harness skills update check-output --source ./updated/check-output --expected CURRENT_REVISION
harness skills update check-output --repository https://github.com/OWNER/REPOSITORY \
  --commit NEW_FULL_COMMIT_ID --path skills/check-output --expected CURRENT_REVISION
harness skills rollback check-output PREVIOUS_REVISION --expected CURRENT_REVISION
harness skills remove check-output --expected CURRENT_REVISION
harness skills rollback check-output PREVIOUS_REVISION --expected absent
```

Every command accepts `--cwd`, `--user`, or an explicit `--root` where applicable.
The default root is `.harness/skills` in the selected workspace. `--user` uses the
active profile's user skill root. Existing configured skill paths can be managed
with `--root`. History is stored outside discovered packages at
`ROOT_PARENT/skill-history/ROOT_NAME/history.db`. Removal retains history.
Existing packages created before history tracking are honestly recorded as an
existing operator-installed package when first updated or removed.

Updates compare the current files with the reviewed digest. An intervening edit
fails the change; the operator must inspect it again. Changes use a persistent
journal and staged directories. A later lifecycle operation completes an
interrupted publication when its files match the recorded states, and refuses
to overwrite conflicting edits. Each update's provenance is retained even when
two source commits have identical contents.

## Learning from execution

Local terminal runs record generic workspace-local evidence when a non-skill
tool actually executed, including when the skill library is initially empty.
Activated skills additionally receive their own usage evidence. Evidence preserves the session ID, activity event IDs,
tool names, outcome, activation hashes when present, and result hashes. Private
prompts, arguments, and tool output are not copied into this ledger. The package
revision is explicitly identified as the revision observed when evidence was
recorded. Repeated recording of the same evidence deduplicates deterministically.
Failed and cancelled runs stay marked as such; no outcome automatically promotes
a new instruction, and execution evidence is not an independent correctness verdict.

```sh
harness skills evidence check-output
harness skills evidence  # completed-work evidence for proposing a new skill
harness skills proposal PROPOSAL_ID
```

The runtime exposes three session-bound tools to local sessions:

- `skill_evidence` reads the latest recorded evidence IDs and outcomes. Omit its
  `name` argument for generic completed-work evidence.
- `skill_propose` cites stored evidence and saves a proposed complete `SKILL.md`,
  explanation, base digest, and full diff. It does not change the installed skill.
  With explicit `create=true`, it proposes a new workspace skill from completed
  work, using an `absent` base and a full new-file diff. This works without any
  previously installed skill.
- `skill_apply` requests workspace-durable approval with the proposal ID, exact
  base digest, and full reviewed diff. It checks proposal/session ownership and
  rejects changed files or altered approval arguments before writing.
  A creation also refuses to overwrite a package that appeared while review was
  pending; the new skill is registered only after its reviewed write succeeds.

Inbox approvals use the existing `harness approvals grant` and
`harness sessions resume` workflow. Explicitly configured approval overrides
still apply. A newly constructed independent local session discovers the updated
skill normally and loads it through `skill_read`; no earlier conversation must be
replayed. Remote conversational sessions cannot record or evolve shared skills.
Existing tips and procedure memory remain separate and available.

Offline regression coverage includes archive attacks and size bounds, update
conflicts, interrupted publication, removal/rollback, duplicate evidence, failed
outcomes, remote exclusion, and a real Agent with a fake adapter that queues an
evolution approval, resumes it once granted, then activates the new instructions
in a fresh independent session. No live model or repository is required by those tests.
