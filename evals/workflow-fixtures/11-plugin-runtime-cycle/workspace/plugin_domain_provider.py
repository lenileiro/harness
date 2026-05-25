from harness.core.domain_profiles import DomainProfile


class DemoProvider:
    def profiles(self):
        return [
            DomainProfile(
                name="plugin-review",
                description="Plugin domain profile for runtime eval",
                allowed_tools=("read_file",),
                system_prompt="PLUGIN REVIEW SYSTEM PROMPT",
            )
        ]
