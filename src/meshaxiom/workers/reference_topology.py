"""Portable v119 topology DiT and frozen 30K EMA topology VAE worker."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time

from meshaxiom.vendor.recipe.runtime import (
    activate_namespace, binding, checkpoint, cuda_device, image_pixels, stage_cli, write_glb, write_json,
)


def _topology_seed(request: dict) -> int:
    import hashlib
    configured = request.get("topology_seed", request["config"]["geometry"].get("topology_seed"))
    if configured is not None:
        if isinstance(configured, bool) or not isinstance(configured, int) or not 0 <= configured < (1 << 63):
            raise ValueError("topology_seed must be an integer in [0,2**63)")
        return configured
    payload = f"meshaxiom-sampling-stream-v1:{request['seed']}:1".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def write_mesh_exports(out: Path, world, exported, header: str) -> None:
    """Serialize cached/native coordinates exactly; GLB supplies the separate Y-up node transform."""
    import numpy as np
    world = np.asarray(world, np.float64).reshape(-1, 3)
    exported = np.asarray(exported, np.int64).reshape(-1, 3)
    with (out / "mesh.obj").open("w") as handle:
        handle.write(f"# {header}\n")
        for x, y, z in world:
            handle.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
        for a, b, c in exported + 1:
            handle.write(f"f {a} {b} {c}\n")
    write_json(out / "mesh.json", {
        "vertices": world.tolist(), "faces": exported.tolist(),
        "up_axis": "z", "display_yaw_degrees": 0,
    })
    write_glb(out / "mesh.glb", world, exported)


def run(request: dict) -> str:
    """Decode only the released rich-8 checkpoint, retaining raw predicted faces separately."""
    import numpy as np
    import torch
    from meshaxiom.workers._geometry_state import load
    from meshaxiom.vendor.recipe import decode_policy90 as decoder, mesh_utils90 as mesh
    state = load(request, "topology", exact=False)
    dev, cfg, model, vae, flow = (state[k] for k in ("dev", "cfg", "model", "vae", "flow"))
    topo_artifact, vae_artifact, actual_basis = (state[k] for k in ("artifact", "vae_artifact", "basis"))
    out = Path(request["output_dir"])
    report_path = out / "geometry_report.json"
    if report_path.exists() or any((out / name).exists() for name in ("mesh.obj", "mesh.json", "mesh.glb", "topology.npz")):
        raise FileExistsError("topology output already exists; use a fresh output directory")
    vertex = json.loads((out / "vertex_report.json").read_text())
    coord_binding = binding(out / "vertices.npz")
    if vertex.get("status") != "COMPLETE" or vertex["coords"]["sha256"] != coord_binding["sha256"]:
        raise ValueError("vertex artifacts do not match their completed report")
    if vertex["checkpoint"]["sha256"] != request["config"]["geometry"]["vertex_checkpoint"]["sha256"]:
        raise ValueError("vertex stage used a different model profile")
    if vertex["sampler"]["seed"] != request["seed"] or vertex["face_budget"] != request["face_budget"]:
        raise ValueError("vertex stage belongs to another seed or face-budget request")
    pixels, input_audit = image_pixels(cfg, request)
    if input_audit["input"]["sha256"] != vertex["image_sha256"]:
        raise ValueError("topology and vertex stages must receive the identical image")
    if input_audit["tensor_sha256"] != vertex["encoder_input_audit"]["tensor_sha256"]:
        raise ValueError("topology and vertex image encoder inputs differ")
    torch.cuda.synchronize(dev)
    started = time.monotonic()
    with np.load(out / "vertices.npz", allow_pickle=False) as archive:
        coords = archive["coords"].astype(np.int64)
    keys = coords[:, 0] * 512 * 512 + coords[:, 1] * 512 + coords[:, 2]
    if coords.shape != (vertex["vertices"], 3) or coords.min() < 0 or coords.max() > 511 or not np.all(np.diff(keys) > 0):
        raise ValueError("vertex artifact no longer satisfies the D9 ordering contract")
    lod = float(2 * len(coords))
    policy = decoder.policy()
    policy["steps"] = request["steps"]
    if policy.get("fill") or policy.get("loops") or policy.get("long_face_k"):
        raise ValueError("the released production policy does not add new hole filling or geometry filters")
    seed = _topology_seed(request)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        context = model.cond(image=pixels.to(dev), camera=None, lod=torch.tensor([lod], dtype=torch.float32, device=dev))
        generator = torch.Generator(device=dev).manual_seed(seed)
        if False:
            with state["post"].patched(stage=False):
                faces, edges, info = state["engine"].decode(model, vae, coords, context, lod, flow, dev, generator, policy)
        else:
            faces, edges, info = decoder.decode(model, vae, coords, context, lod, flow, dev, generator, policy)
    torch.cuda.synchronize(dev)
    faces = np.asarray(faces, np.int64).reshape(-1, 3)
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    world = (coords.astype(np.float64) + 0.5) / 512 * 2 - 1
    if len(faces) == 0 or faces.min() < 0 or faces.max() >= len(world) or not np.isfinite(world).all():
        raise ValueError("decoder produced an empty or invalid mesh")
    stats = mesh.mesh_stats(world, faces)
    stats["stored_faces"] = len(faces)
    exported, orientation = mesh.export_faces(world, faces)
    orientation["applied_to"] = ["mesh.obj", "mesh.json", "mesh.glb"]
    header = f"MeshAxiom vertex {vertex['step']} EMA + topology {topo_artifact['step']} EMA / VAE {vae_artifact['step']} EMA"
    write_mesh_exports(out, world, exported, header)
    np.savez_compressed(out / "topology.npz", coords=coords.astype(np.int16), faces=faces.astype(np.int32),
                        edges=edges.astype(np.int32), export_faces=exported.astype(np.int32))
    artifacts = {name: binding(out / filename) for name, filename in {
        "vertices": "vertices.npz", "obj": "mesh.obj", "mesh_json": "mesh.json", "mesh_glb": "mesh.glb", "topology": "topology.npz"
    }.items()}
    report = {
        "status": "COMPLETE", "stage": "geometry", "vertex_checkpoint": vertex["checkpoint"],
        "implementation": "reference", "models_resident": True,
        "topology_checkpoint": topo_artifact, "vae_checkpoint": vae_artifact,
        "actual_checkpoint_steps": {"vertex": vertex["step"], "topology": topo_artifact["step"], "vae": vae_artifact["step"]},
        "weights": {"vertex": "ema", "topology": "ema", "vae": "ema"},
        "vae_basis_sha256": actual_basis, "image_sha256": input_audit["input"]["sha256"],
        "encoder_input_audit": input_audit, "vertex_report": binding(out / "vertex_report.json"),
        "face_budget": request["face_budget"], "face_budget_semantics": "vertex condition request; actual faces are predicted, not forcibly decimated",
        "sampler": {"vertex": vertex["sampler"], "topology": policy},
        "seed": request["seed"], "topology_seed": seed, "topology_lod": lod,
        "production_recipe": request["steps"] == 20,
        "decode_info": info, "stats": stats, "export_orientation": orientation,
        "coordinate_frame": "z_up", "coordinate_grid": {"depth": 9, "world": "(coords+0.5)/512*2-1"},
        "artifact_coordinate_frames": {
            "vertices": "z_up_integer_D9", "topology": "z_up_integer_D9",
            "obj": "z_up", "mesh_json": "z_up",
            "mesh_glb": {"vertex_storage": "z_up", "scene_world": "y_up",
                         "node_rotation_xyzw": [-(0.5 ** 0.5), 0.0, 0.0, 0.5 ** 0.5],
                         "scene_transform": "(x,y,z) -> (x,z,-y)"},
        },
        "raw_faces_preserved": True, "export_only_winding": True, "gt_used": False,
        "seconds": round(time.monotonic() - started, 3), "artifacts": artifacts,
    }
    write_json(report_path, report)
    return str(report_path)


if __name__ == "__main__":
    stage_cli(run)
