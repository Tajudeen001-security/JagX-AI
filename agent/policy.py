from dataclasses import dataclass

@dataclass(frozen=True)
class ToolPolicy:
    allow_files: bool = True
    allow_shell: bool = False
    allow_network: bool = False
    allow_deploy: bool = False
    allow_security_testing: bool = False

    def can_use(self, capability: str) -> bool:
        return {
            "files": self.allow_files,
            "shell": self.allow_shell,
            "network": self.allow_network,
            "deploy": self.allow_deploy,
            "security_testing": self.allow_security_testing,
        }.get(capability, False)
