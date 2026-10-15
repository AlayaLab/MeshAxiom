"""Single-GPU Space with one persistent Exact pipeline and a serial queue."""
import atexit
import os
from pathlib import Path
import threading
import uuid

import gradio as gr

from meshaxiom import MeshAxiomPipeline

_PIPE = None
_LOCK = threading.Lock()


def generate(image, face_budget, background):
    global _PIPE
    if image is None:
        raise gr.Error("Upload an image first.")
    with _LOCK:
        if _PIPE is None:
            local = os.environ.get("MESHAXIOM_MODEL_DIR")
            repo = os.environ.get("MESHAXIOM_MODEL_ID")
            if not local and not repo:
                raise gr.Error("Configure MESHAXIOM_MODEL_DIR or MESHAXIOM_MODEL_ID and MESHAXIOM_MODEL_REVISION.")
            _PIPE = MeshAxiomPipeline.from_pretrained(
                local or repo, revision=os.environ.get("MESHAXIOM_MODEL_REVISION"),
                token=os.environ.get("HF_TOKEN"), device="cuda:0",
            )
        root = Path(os.environ.get("MESHAXIOM_OUTPUT_DIR", "/tmp/meshaxiom-outputs"))
        result = _PIPE.generate(image, root / uuid.uuid4().hex, face_budget=int(face_budget), background=background)
        files = [result["artifacts"][k]["path"] for k in ["obj", "mesh_json", "mesh_glb", "vertices", "topology"]]
        return result["artifacts"]["mesh_glb"]["path"], files


def close():
    if _PIPE is not None:
        _PIPE.close()


atexit.register(close)
with gr.Blocks(title="MeshAxiom") as demo:
    gr.Markdown("# MeshAxiom\nGenerate a mesh from an image with Exact inference.")
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="filepath", image_mode="RGBA", label="Input image")
            face_budget = gr.Dropdown([2000, 4000, 8000, 16000], value=4000, label="Face request")
            background = gr.Dropdown(["auto", "alpha", "keep", "remove"], value="auto", label="Background")
            button = gr.Button("Generate", variant="primary")
        with gr.Column():
            preview = gr.Model3D(label="Generated mesh")
            files = gr.File(file_count="multiple", label="Download")
    gr.Markdown("The face request guides generation; actual face counts may differ. The first request also loads the models.")
    button.click(generate, [image, face_budget, background], [preview, files], concurrency_limit=1, api_name="generate_mesh")

if __name__ == "__main__":
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")),
        allowed_paths=[os.environ.get("MESHAXIOM_OUTPUT_DIR", "/tmp/meshaxiom-outputs")],
    )
