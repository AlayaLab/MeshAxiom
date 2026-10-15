# Exact inference

Only Exact is exposed by generation, Python, JSON tools, HTTP and the Space UI.
The model recipe is fixed: vertex EMA 180K, topology EMA 10K, VAE EMA 30K;
MBR8 at select level 3, vertex CFG 3×4/5×5, DPM++2M with 20 steps, topology CFG 3,
and topology seed 1574365204. The vertex seed defaults to 2026082301.

The implementation integrates Sho Kuno's exact-shape CUDA graphs for all vertex
levels, separate conditional/unconditional graph replays and cached context K/V.
The original duplicated 2050-token image context is preserved. Topology forwards
use exact-length graphs; vectorized extraction and orientation retain the original
geometry decisions. The serialized tensors are not quantized or distilled.
Linear layers have the same bf16 values used by CUDA autocast; the checkpoint
values and other fp32 parameters are preserved.

The two frozen model code trees are isolated in separate persistent processes.
This avoids namespace collisions while retaining both models between requests.
The parent serializes requests and holds the GPU lease while the models are
resident, including the interval between calls. Worker failures and finite
request timeouts close both workers; a partially failed CUDA context is not reused.
Material operations first close the geometry workers to release their memory.

The packaged reference path is a verification diagnostic, with the same local
model construction, preparation and output formatting as Exact. `verify-exact`
requires the SHA256 of all four files to match, and returns failure otherwise.
Rebranding changes the OBJ header; comparisons use the newly branded reference,
not an old file whose header contains the former product name.

Source provenance hashes are in `vendor/kuno/ORIGIN.json`. Original source trees
and raw colleague evidence are retained in private preparation records, outside
the publication payload. Module/display names and imports have been adapted;
no extra face simplification, hole filling or quality-based selection is added.

Existing author evidence reports four-input file identity and approximately 1.6×
median acceleration over 400 development requests at a 4K request on H200. Those
measurements cover sampling and decoding with models resident and exclude loading.
They describe the original delivered implementation. The integration's own
validation status is recorded in `VALIDATION.json`; CPU checks do not establish
CUDA-graph identity or new latency measurements.
