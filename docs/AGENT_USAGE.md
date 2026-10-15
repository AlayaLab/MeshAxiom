# Agent interfaces

All interfaces call the same pipeline and expose three tools:

- `generate_mesh(image, output_dir, face_budget, seed, background)` uses Exact.
- `texture_mesh(mesh, image, output_dir, seed, texture_size, camera)` textures an existing mesh independently.
- `enhance_texture(image, output_dir)` performs independent 2× enhancement.

```bash
meshaxiom run-tool --config models/MeshAxiom-Base/meshaxiom.json --request examples/generate.json
meshaxiom serve --config models/MeshAxiom-Base/meshaxiom.json --workspace outputs/server
```

The local server binds to loopback. `GET /agent-tools.json` and
`GET /openapi.json` expose the schemas. Upload a file to `POST /v1/uploads` with
`X-Filename`; submit its returned token to `POST /v1/jobs`; poll
`GET /v1/jobs/{id}` and download completed artifacts from the URLs in the result.
For a geometry job:

```json
{"operation":"geometry","image_upload":"<returned-upload-id>","face_budget":4000,"background":"auto"}
```

The queue is bounded and serial; one pipeline is retained across jobs. Every
job has a fresh output directory. `status.json`, `request.json`, `result.json`
and failure records retain the job parameters and file SHA256 bindings.
Generation interfaces do not accept an implementation selector.

For Python, keep a `MeshAxiomPipeline` in a context manager across requests.
`MeshAxiomPipeline.from_pretrained` accepts a local directory or an actual
published Hub ID with a full commit SHA. It downloads data only and never executes
remote model code. The Docker Space separately exposes the named Gradio endpoint
`generate_mesh`; its URL and protocol are provided by that deployed Space.
