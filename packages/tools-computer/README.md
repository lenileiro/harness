# Local desktop tools

`ComputerToolset` controls the primary local desktop through PyAutoGUI. It is
disabled by default and requires the optional desktop dependencies:

```sh
uv sync --package tools-computer --extra desktop
```

```python
from pathlib import Path
from harness.tools.computer import ComputerConfig, ComputerToolset

async with ComputerToolset(ComputerConfig(enabled=True), cwd=Path.cwd()) as desktop:
    # Register desktop.tools with the agent's managed tool context.
    pass
```

The `computer` tool supports screenshot, position, click, move, drag, type,
press, hotkey and vertical scroll. Every action uses Harness approval policy.
Coordinates refer to the primary screen: Retina screenshots are resized to the
same coordinate dimensions used for input. PNG screenshots become model-visible
image attachments and unique files under `artifacts/computer`. Paths exclude
Harness private state, symlinks and traversal; screenshot bytes are capped.

A per-user file lock prevents separate Harness processes from simultaneously
controlling the desktop. PyAutoGUI's corner fail-safe stays enabled during the
context. Calls are serialized; cancellation waits for an issued OS action to
finish before releasing ownership. Text is sent in small chunks so cancellation
stops further typing. Toolsets close with their managed agent context.

This is a full local GUI capability, so remote gateway users must not receive
it. It requires an existing graphical session and OS accessibility/screen-capture
permissions. PyAutoGUI currently targets the primary monitor; this implementation
supports ASCII typing, named keys, and hotkeys. It does not provide Unicode
clipboard paste, OCR, secondary-monitor targeting or a virtual desktop.

Tests use a fake driver and valid generated PNG fixtures to verify dispatch,
coordinate normalization, approval classification, artifact boundaries,
exclusive ownership and cancellation. No live desktop was operated during
implementation, so each target OS still needs an explicit GUI integration check.
The backend API follows the [official PyAutoGUI documentation](https://pyautogui.readthedocs.io/en/latest/).
