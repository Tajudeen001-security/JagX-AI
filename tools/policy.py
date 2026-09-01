from dataclasses import dataclass

@dataclass(frozen=True)
class ToolPolicy:
    filesystem: bool=False
    shell: bool=False
    network: bool=False
    deployment: bool=False
    security_testing: bool=False

    def allows(self,name:str)->bool:
        return {
            'filesystem':self.filesystem,
            'shell':self.shell,
            'network':self.network,
            'deployment':self.deployment,
            'security_testing':self.security_testing,
        }.get(name,False)

class PolicyError(PermissionError): pass

def require(policy:ToolPolicy,name:str):
    if not policy.allows(name): raise PolicyError(f'tool permission denied: {name}')
