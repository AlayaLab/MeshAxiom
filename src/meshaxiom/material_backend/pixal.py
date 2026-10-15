"""Portable, texture-only calls into separately installed official Pixal3D."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .geometry import file_sha256


def _directory(settings, name):
    if not settings.get(name):
        raise ValueError(f"config.material.{name} must name an existing local model/source directory")
    path = Path(settings[name]).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"config.material.{name} is not a directory")
    return path


def validate_material_config(config):
    """Preflight local artifacts on CPU before importing a CUDA package."""
    settings = config.get("material", {})
    directories = {name: _directory(settings, name) for name in (
        "pixal_model_dir", "trellis2_model_dir", "image_encoder_dir", "naf_repo")}
    if not (directories["naf_repo"] / "hubconf.py").is_file():
        raise ValueError("NAF source directory does not contain hubconf.py")
    model_prefixes = {
        "shape_slat_encoder": directories["trellis2_model_dir"] / "ckpts/shape_enc_next_dc_f16c32_fp16",
        "tex_slat_decoder": directories["trellis2_model_dir"] / "ckpts/tex_dec_next_dc_f16c32_fp16",
        "tex_slat_flow_model_1024": directories["pixal_model_dir"] / "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
    }
    for name, prefix in model_prefixes.items():
        for suffix in (".json", ".safetensors"):
            if not Path(str(prefix) + suffix).is_file():
                raise FileNotFoundError(f"Missing local {name} artifact: {prefix}{suffix}")
    pipeline_path = directories["pixal_model_dir"] / "pipeline.json"
    pipeline = json.loads(pipeline_path.read_text())["args"]
    for field in ("tex_slat_sampler", "shape_slat_normalization", "tex_slat_normalization"):
        if field not in pipeline:
            raise ValueError(f"Pixal3D pipeline.json is missing {field}")
    for filename in ("config.json",):
        if not (directories["image_encoder_dir"] / filename).is_file():
            raise FileNotFoundError("Image encoder directory is not a complete local Hugging Face snapshot")
    encoder_weights = sorted(directories["image_encoder_dir"].glob("*.safetensors"))
    if not encoder_weights:
        raise FileNotFoundError("No local DINOv3 safetensors weights found")
    naf_checkpoint = settings.get("naf_checkpoint")
    if not naf_checkpoint:
        # torch.hub's default checkpoint location, without importing torch.
        torch_home = Path(os.environ.get("TORCH_HOME", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "torch"))
        naf_checkpoint = torch_home / "hub/checkpoints/naf_release.pth"
    naf_checkpoint = Path(naf_checkpoint).expanduser().resolve(strict=True)
    if not naf_checkpoint.is_file():
        raise FileNotFoundError("Local NAF weights are required; set config.material.naf_checkpoint")
    return settings, directories, model_prefixes, pipeline_path, pipeline, encoder_weights, naf_checkpoint


def _source_binding(module, repository=None):
    filename = Path(module.__file__).resolve()
    result = {"file": str(filename), "sha256": file_sha256(filename)}
    if repository:
        completed = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if completed.returncode == 0:
            result["git_commit"] = completed.stdout.strip()
    return result


def texture_mesh(config, aligned_vertices, faces, reference, camera, output_dir, *, seed=42, texture_size=2048, device="cuda:0"):
    """Encode the supplied mesh; sample only texture latents; bake its triangles."""
    settings, directories, prefixes, pipeline_path, pipeline_config, encoder_weights, naf_checkpoint = validate_material_config(config)
    repo = settings.get("pixal_repo")
    if repo:
        repo = Path(repo).expanduser().resolve(strict=True)
        if not (repo / "pixal3d").is_dir():
            raise ValueError("config.material.pixal_repo must be an official source checkout containing pixal3d/")
        sys.path.insert(0, str(repo))
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("ATTN_BACKEND", str(settings.get("attention_backend", "sdpa")))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # A writable per-run GEMM cache never alters a model or external checkout.
    cache = Path(output_dir) / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    autotune_cache = cache / "autotune_cache.json"
    if not autotune_cache.exists():
        if repo and (repo / "autotune_cache.json").is_file():
            import shutil
            shutil.copyfile(repo / "autotune_cache.json", autotune_cache)
        else:
            autotune_cache.write_text("{}\n")
    os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = str(autotune_cache)

    # Full hashes bind actual weights, not a filename that resembles a hash.
    print("Binding local Pixal3D/TRELLIS.2/DINOv3/NAF artifacts", flush=True)
    weight_bindings = {}
    for name, prefix in prefixes.items():
        weight_bindings[name] = {
            "config_path": str(prefix) + ".json", "config_sha256": file_sha256(str(prefix) + ".json"),
            "weights_path": str(prefix) + ".safetensors", "weights_sha256": file_sha256(str(prefix) + ".safetensors"),
        }
    weight_bindings["image_encoder"] = {"model_directory": str(directories["image_encoder_dir"]),
        "config_sha256": file_sha256(directories["image_encoder_dir"] / "config.json"),
        "weights": [{"path": str(path), "sha256": file_sha256(path)} for path in encoder_weights]}
    weight_bindings["naf"] = {"path": str(naf_checkpoint), "sha256": file_sha256(naf_checkpoint), "hubconf_sha256": file_sha256(directories["naf_repo"] / "hubconf.py")}

    import torch
    import numpy as np
    import trimesh
    from safetensors import safe_open
    from pixal3d import models
    from pixal3d.pipelines import samplers
    from pixal3d.pipelines import pixal3d_image_to_3d, trellis2_texturing
    from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj

    selected_device = torch.device(device)
    if selected_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Pixal3D texture inference requires a CUDA device; CPU tests do not run this backend")
    if selected_device.index is None:
        selected_device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(selected_device)
    torch.set_num_threads(int(settings.get("cpu_threads", 2)))
    fraction = settings.get("memory_fraction")
    if fraction is not None:
        if not 0 < float(fraction) <= 1:
            raise ValueError("material.memory_fraction must lie in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(float(fraction), selected_device)
    torch.cuda.reset_peak_memory_stats(selected_device)
    started = time.monotonic()
    loaded, key_checks = {}, {}
    with torch.inference_mode():
        for name, prefix in prefixes.items():
            print("Loading", name, flush=True)
            model = models.from_pretrained(str(prefix)).eval().requires_grad_(False)
            with safe_open(str(prefix) + ".safetensors", framework="pt", device="cpu") as stored:
                stored_keys = set(stored.keys())
                state = model.state_dict()
                if stored_keys != set(state):
                    raise RuntimeError(f"Official {name} loader did not load an equivalent complete state dict")
                if any(tuple(stored.get_slice(key).get_shape()) != tuple(state[key].shape) for key in stored_keys):
                    raise RuntimeError(f"Checkpoint tensor shape mismatch for {name}")
            loaded[name] = model
            key_checks[name] = {"state_keys_and_shapes_exact": True, "keys": len(stored_keys)}
        sampler = pipeline_config["tex_slat_sampler"]
        pipe = pixal3d_image_to_3d.Pixal3DImageTo3DPipeline(loaded,
            tex_slat_sampler=getattr(samplers, sampler["name"])(**sampler["args"]),
            tex_slat_sampler_params=sampler["params"],
            shape_slat_normalization=pipeline_config["shape_slat_normalization"],
            tex_slat_normalization=pipeline_config["tex_slat_normalization"], low_vram=True)
        pipe.to(selected_device)
        mesh = trimesh.Trimesh(vertices=np.asarray(aligned_vertices).copy(), faces=np.asarray(faces).copy(), process=False)
        print("Encoding supplied mesh support at 1024", flush=True)
        shape = trellis2_texturing.Trellis2TexturingPipeline.encode_shape_slat(pipe, mesh, 1024)
        condition_model = image_conditioned_proj.DinoV3ProjFeatureExtractor(str(directories["image_encoder_dir"]),
            image_size=1024, grid_resolution=64, use_naf_upsample=True, naf_target_size=1024).eval()
        # Supplying the local model disables upstream's implicit Hub download.
        naf = torch.hub.load(str(directories["naf_repo"]), "naf", source="local", pretrained=False, device="cpu")
        naf.load_state_dict(torch.load(naf_checkpoint, map_location="cpu", weights_only=True), strict=True)
        condition_model.naf_model = naf.eval().requires_grad_(False)
        condition = pipe.get_proj_cond_shape(condition_model, [reference], shape.coords,
            camera_angle_x=camera["camera_angle_x"], distance=camera["distance"], mesh_scale=1.)
        del condition_model, naf
        gc.collect(); torch.cuda.empty_cache()
        torch.manual_seed(int(seed)); torch.cuda.manual_seed_all(int(seed))
        print("Sampling texture latents only", flush=True)
        texture = pipe.sample_tex_slat(condition, loaded["tex_slat_flow_model_1024"], shape)
        del condition
        gc.collect(); torch.cuda.empty_cache()
        if not hasattr(shape, "_spatial_cache") or not hasattr(shape, "_scale"):
            raise RuntimeError("Installed sparse tensor API cannot preserve the encoder's original support")
        texture._spatial_cache = shape._spatial_cache
        texture._scale = shape._scale
        pbr = trellis2_texturing.Trellis2TexturingPipeline.decode_tex_slat(pipe, texture)
        print("UV-baking the original triangles", flush=True)
        baked = trellis2_texturing.Trellis2TexturingPipeline.postprocess_mesh(pipe, mesh, pbr, 1024, int(texture_size))
        provenance = {
            "backend": "Pixal3D official texture flow + TRELLIS.2 supplied-shape encoder/PBR decoder",
            "resolution": 1024, "texture_size": int(texture_size), "seed": int(seed), "device": str(selected_device),
            "tex_slat_sampler": sampler, "shape_tokens": int(shape.coords.shape[0]), "pbr_voxels": int(pbr.coords.shape[0]),
            "shape_support_cache_reused": True, "shape_generated": False, "remeshed": False,
            "weight_bindings": weight_bindings, "weight_key_checks": key_checks,
            "pipeline_config_sha256": file_sha256(pipeline_path),
            "source": {"pixal_pipeline": _source_binding(pixal3d_image_to_3d, repo),
                "texture_pipeline": _source_binding(trellis2_texturing, repo), "projection": _source_binding(image_conditioned_proj, repo)},
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(selected_device), "seconds": time.monotonic() - started,
        }
    return baked, provenance
