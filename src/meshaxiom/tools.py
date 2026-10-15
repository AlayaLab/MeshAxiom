"""Function-tool definitions and local dispatch for agent frameworks."""
import json
from pathlib import Path
from .contracts import MeshAxiomError

def tool_definitions():
    return json.loads(Path(__file__).with_name("schemas").joinpath("agent-tools.json").read_text())["tools"]

def execute_tool(pipeline, call):
    if not isinstance(call, dict) or set(call) != {"name", "arguments"} or not isinstance(call["arguments"], dict):
        raise MeshAxiomError("Tool call must contain name and an arguments object")
    name, arguments = call["name"], dict(call["arguments"])
    if not isinstance(name, str) or name not in {"generate_mesh", "texture_mesh", "enhance_texture"}:
        raise MeshAxiomError("Unknown tool name")
    required = {"image", "output_dir"} | ({"mesh"} if name == "texture_mesh" else set())
    optional = {"seed", "face_budget", "background"} if name == "generate_mesh" else {"seed", "texture_size", "camera"} if name == "texture_mesh" else set()
    if required - set(arguments) or set(arguments) - required - optional:
        raise MeshAxiomError("Unknown tool or invalid tool arguments")
    if any(not isinstance(arguments[key], str) or not arguments[key] for key in required):
        raise MeshAxiomError("Tool input and output paths must be nonempty strings")
    if name == "generate_mesh":
        return pipeline.generate(**arguments)
    if name == "enhance_texture":
        return pipeline.enhance(**arguments)
    return pipeline.texture(**arguments)

