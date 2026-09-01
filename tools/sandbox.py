from pathlib import Path

class WorkspaceSandbox:
    def __init__(self,root): self.root=Path(root).resolve(); self.root.mkdir(parents=True,exist_ok=True)
    def path(self,relative):
        target=(self.root/relative).resolve()
        if target!=self.root and self.root not in target.parents: raise ValueError("Path escapes sandbox")
        return target
    def write(self,relative,content):
        target=self.path(relative); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(content,encoding="utf-8")
    def read(self,relative): return self.path(relative).read_text(encoding="utf-8")
