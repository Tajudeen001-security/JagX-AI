from __future__ import annotations

from pathlib import Path


PROJECT_GODOT = '''[application]
config/name="{name}"
run/main_scene="res://main.tscn"

[display]
window/size/viewport_width={width}
window/size/viewport_height={height}

[rendering]
renderer/rendering_method="gl_compatibility"
renderer/rendering_method.mobile="gl_compatibility"
'''

MAIN_TSCN = '''[gd_scene load_steps=2 format=3]

[node name="Main" type="Node2D"]

[node name="Label" type="Label" parent="."]
offset_left = 40.0
offset_top = 40.0
offset_right = 400.0
offset_bottom = 80.0
text = "JagX Godot project"
'''

MAIN_GD = '''extends Node2D

func _ready() -> void:
\tprint("JagX Godot project ready")
'''


def generate_godot_project(
    root: str | Path,
    *,
    name: str = "JagX Generated Game",
    width: int = 1280,
    height: int = 720,
) -> Path:
    """Create a minimal Godot 4.x project structure on disk."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.godot").write_text(
        PROJECT_GODOT.format(name=name, width=width, height=height), encoding="utf-8"
    )
    (root / "main.tscn").write_text(MAIN_TSCN, encoding="utf-8")
    (root / "main.gd").write_text(MAIN_GD, encoding="utf-8")
    return root
