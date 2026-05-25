# Create and validate a workspace experience plugin

Use the workspace plugin mechanism in this fixture directory to add one
experience provider plugin, then validate it through the real Harness CLI.

Do not edit files under the repo root. Work only inside this copied fixture
workspace.

Create these two files:

1. `.harness/plugins/workspace-experience.toml`
2. `workspace_plugin.py`

Requirements:

- The manifest must declare:
  - `name = "workspace-experience"`
  - `kind = "experience"`
  - `provider = "workspace_plugin:WorkspaceExperienceProvider"`
- `workspace_plugin.py` must define `WorkspaceExperienceProvider`
- Use this import:
  - `from harness.core.tips_models import Tip`
- `WorkspaceExperienceProvider.query()` should return exactly one tip when the
  task text mentions `plugin`
- That tip text must be:
  `Use plugin validation before runtime experiments.`
- For unrelated task text, return no tips

Use this exact provider shape:

```python
from harness.core.tips_models import Tip


class WorkspaceExperienceProvider:
    def query(self, task_text: str, *, top_k: int = 3):
        if "plugin" not in task_text.lower():
            return []
        return [
            Tip(
                text="Use plugin validation before runtime experiments.",
                triggers=("plugin",),
                weight=2.0,
            )
        ]
```

After creating the files, run:

- `harness plugins validate --kind experience`

Then run the tests.

Do not search the repo for other provider implementations. The exact pattern
above is sufficient.
