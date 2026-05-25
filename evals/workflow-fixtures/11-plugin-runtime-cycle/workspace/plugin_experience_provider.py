from harness.core.tips_models import Tip


class DemoProvider:
    def query(self, task_text: str, *, top_k: int = 3):
        return [
            Tip(
                text="plugin guidance from experience provider",
                triggers=("review",),
                weight=5.0,
            )
        ]
