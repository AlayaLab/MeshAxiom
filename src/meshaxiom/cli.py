"""Human CLI and machine-readable command interface."""
import argparse
import json
import sys
from pathlib import Path
from .config import load_config, doctor
from .contracts import MeshAxiomError
from .pipeline import MeshAxiomPipeline

def parser():
    p = argparse.ArgumentParser(prog="meshaxiom", description="Generate native meshes from an image, or texture an existing mesh.")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["generate", "texture", "enhance", "doctor", "serve", "run-tool", "verify-exact"]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True, help="JSON model/runtime configuration")
        if name in {"generate", "verify-exact"}:
            cmd.add_argument("--image", required=True)
            cmd.add_argument("--output", required=True)
            cmd.add_argument("--face-budget", type=int, default=4000 if name == "verify-exact" else 8000)
            cmd.add_argument("--seed", type=int, default=2026082301)
            cmd.add_argument("--background", choices=["auto", "alpha", "keep", "remove"], default="keep" if name == "verify-exact" else "auto")
        elif name == "texture":
            cmd.add_argument("--mesh", required=True)
            cmd.add_argument("--image", required=True)
            cmd.add_argument("--output", required=True)
            cmd.add_argument("--texture-size", type=int, default=2048)
            cmd.add_argument("--seed", type=int, default=42)
            cmd.add_argument("--camera", help="Camera JSON file; omitted to fit the reference foreground")
        elif name == "enhance":
            cmd.add_argument("--image", required=True)
            cmd.add_argument("--output", required=True)
        elif name == "doctor":
            cmd.add_argument("--operation", choices=["all", "geometry", "material", "superresolution"], default="all")
            cmd.add_argument("--verify-hashes", action="store_true")
            cmd.add_argument("--runtime-check", action="store_true", help="Import runtime dependencies without loading model weights")
        elif name == "run-tool":
            cmd.add_argument("--request", default="-", help="Tool-call JSON file, or - for stdin")
        else:
            cmd.add_argument("--port", type=int, default=7860)
            cmd.add_argument("--workspace", default="outputs/server")
    return p

def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "doctor":
            result = doctor(config, args.operation, verify_hashes=args.verify_hashes, check_runtime=args.runtime_check)
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result["status"] == "ready" else 2
        if args.command == "serve":
            from .server import serve
            serve(config, args.workspace, port=args.port)
            return 0
        if args.command == "verify-exact":
            from .verification import verify_exact
            result = verify_exact(config, args.image, args.output, face_budget=args.face_budget, seed=args.seed, background=args.background)
            print(json.dumps(result, ensure_ascii=False))
            return 0
        pipeline = MeshAxiomPipeline(config)
        try:
            if args.command == "run-tool":
                from .tools import execute_tool
                request = json.load(sys.stdin) if args.request == "-" else json.loads(Path(args.request).read_text())
                result = execute_tool(pipeline, request)
            elif args.command == "generate":
                result = pipeline.generate(args.image, args.output, face_budget=args.face_budget, seed=args.seed, background=args.background)
            elif args.command == "enhance":
                result = pipeline.enhance(args.image, args.output)
            else:
                camera = json.loads(Path(args.camera).read_text()) if args.camera else None
                result = pipeline.texture(args.mesh, args.image, args.output, texture_size=args.texture_size, seed=args.seed, camera=camera)
        finally:
            pipeline.close()
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (MeshAxiomError, OSError, ValueError) as exc:
        print(json.dumps({"schema_version": 1, "status": "failed", "error": str(exc), "error_type": type(exc).__name__}, ensure_ascii=False), file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

