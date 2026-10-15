"""CPU evidence for integration contracts; these tests do not claim CUDA identity."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from meshaxiom.config import load_config
from meshaxiom.contracts import MeshAxiomError
from meshaxiom.hub import download_model
from meshaxiom.session import ResidentWorker
from meshaxiom.verification import compare_outputs

ROOT = Path(__file__).resolve().parents[1]


def test_no_other_implementation_is_accepted():
    with pytest.raises(MeshAxiomError, match="Only the Exact"):
        load_config({"schema_version": 1, "geometry": {"implementation": "other"}})


def test_model_download_requires_immutable_revision():
    with pytest.raises(MeshAxiomError, match="immutable"):
        download_model("AlayaLab/MeshAxiom-Base")
    with pytest.raises(MeshAxiomError, match="40-character"):
        download_model("AlayaLab/MeshAxiom-Base", revision="main")


def test_identity_requires_all_four_real_file_hashes(tmp_path):
    for directory in [tmp_path / "reference", tmp_path / "exact"]:
        directory.mkdir()
        for name in ["vertices.npz", "topology.npz", "mesh.obj", "mesh.json"]:
            (directory / name).write_bytes(b"actual bytes:" + name.encode())
    assert compare_outputs(tmp_path / "reference", tmp_path / "exact")["status"] == "PASS"
    (tmp_path / "exact/mesh.json").write_text("different")
    assert compare_outputs(tmp_path / "reference", tmp_path / "exact")["status"] == "FAIL"
    (tmp_path / "exact/mesh.json").unlink()
    assert compare_outputs(tmp_path / "reference", tmp_path / "exact")["status"] == "FAIL"


def test_real_resident_process_failure_is_reported_and_cleaned_up(tmp_path):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"not_a_model_request": True}))
    worker = ResidentWorker("vertex", {"vertex_python": sys.executable, "timeout_seconds": 20,
                                       "scratch_directory": str(tmp_path)})
    with pytest.raises(MeshAxiomError, match="config object"):
        worker.run(request, tmp_path)
    assert worker.process is None
    assert worker._reply is None
    assert "config object" in (tmp_path / "vertex.log").read_text()


def cpu_script(code, timeout=90):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    process = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                             text=True, timeout=timeout)
    assert process.returncode == 0, process.stdout + process.stderr


def test_real_sampler_control_flow_matches_packaged_reference_on_cpu():
    cpu_script('''
import sys
from pathlib import Path
import torch, numpy as np
from meshaxiom.vendor.recipe.runtime import activate_namespace
activate_namespace("vertex")
from meshaxiom_core.models.octree_dit import MeshAxiomDiT
from meshaxiom_core.diffusion.flow import RectifiedFlow
from meshaxiom.vendor.recipe.gen_mbr128 import sample_mbr as reference
from meshaxiom.vendor.kuno.p4_vertex_engine import sample_mbr, EagerBackend
torch.set_num_threads(2)
class Condition(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.register_buffer("null", torch.zeros(1,3,32))
    def null_ctx(self,b): return self.null.expand(b,-1,-1)
class VM(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.depth_total=4; self.target_mode="unit"
        self.cond=Condition()
        self.dit=MeshAxiomDiT(8,8,dim=48,depth=2,heads=4,ctx_dim=32,max_level=4)
torch.manual_seed(91); vm=VM().eval().requires_grad_(False); flow=RectifiedFlow()
ctx=torch.randn(1,6,32)
for seed in [0,42,2026082301]:
    options=dict(K=8,select_level=3,seed=seed,thr=.4,cfg=[3.,3.,3.,5.],steps=20,sampler="dpmpp2m",max_nodes=64)
    with torch.inference_mode():
        a,ai=reference(vm,ctx,flow,**options)
        b,bi=sample_mbr(EagerBackend(vm,flow),vm,ctx,**options)
    assert len(a)==len(b) and all(np.array_equal(x,y) for x,y in zip(a,b))
    assert ai==bi, (ai,bi)
print("CPU MBR8 reference/control-flow arrays identical for 3 seeds")
''')


def test_vectorized_orientation_and_capacity_match_reference_on_cpu():
    cpu_script('''
import torch,numpy as np
from meshaxiom.vendor.recipe.runtime import activate_namespace
activate_namespace("topology")
from meshaxiom.vendor.recipe import decode_policy90 as dp,mesh_utils90 as mu
from meshaxiom.vendor.kuno import p5_post_fast as post
torch.set_num_threads(2);rng=np.random.default_rng(101)
for n in [5,16,50]:
    vertices=rng.normal(size=(n,3))
    faces=np.array([rng.choice(n,3,replace=False) for _ in range(n*3)],np.int64)
    a,_=mu.export_faces(vertices,faces); b,_=post.export_faces_fast(vertices,faces)
    assert np.array_equal(a,b)
    z=torch.tensor(rng.normal(size=(n,16)),dtype=torch.float32)
    for cap in [1,2,3]:
        a=dp.prune_edge_capacity(z,faces,"spacetime",cap=cap)
        b=post.prune_edge_capacity_fast(z,faces,"spacetime",cap=cap)
        assert np.array_equal(a,b)
print("CPU orientation/capacity arrays identical")
''')


def test_exact_static_buffers_and_kv_match_eager_cpu_control():
    cpu_script('''
import sys
from pathlib import Path
import torch,numpy as np
from meshaxiom.vendor.recipe.runtime import activate_namespace
activate_namespace("vertex")
from meshaxiom_core.models.octree_dit import MeshAxiomDiT
from meshaxiom_core.diffusion.flow import RectifiedFlow
from meshaxiom.vendor.kuno import vertex_engine as nf
sys.modules["vertex_engine"]=nf
from meshaxiom.vendor.kuno.p7_exact_backend import ExactBackend
from meshaxiom.vendor.kuno.p4_vertex_engine import EagerBackend,sample_mbr
torch.set_num_threads(2)
class Condition(torch.nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer("null",torch.zeros(1,3,32))
    def null_ctx(self,b):return self.null.expand(b,-1,-1)
class VM(torch.nn.Module):
    def __init__(self):
        super().__init__();self.depth_total=4;self.target_mode="unit";self.cond=Condition()
        self.dit=MeshAxiomDiT(8,8,dim=48,depth=2,heads=4,ctx_dim=32,max_level=4)
class CPUStaticSlot:
    # Real model/KV arithmetic on CPU static buffers, with no CUDA capture claim.
    def __init__(self,dit,B,N,ctx_shape,device,kvcache,masked=True,warmup=0):
        self.dit=dit;self.masked=masked;self.capture_s=0.
        self.x=torch.zeros(B,N,8);self.t=torch.zeros(B);self.pos=torch.zeros(B,N,3)
        self.ctx=torch.zeros(B,*ctx_shape);self.lengths=torch.full((B,),N,dtype=torch.long)
        self.level=torch.ones(B,dtype=torch.long)
    def fill_kv(self,ctx):self.kv=[nf.compute_kv(block.cross,ctx) for block in self.dit.blocks]
    def capture(self):pass
    def replay(self):
        nf.clear_kv(self.dit)
        for block,kv in zip(self.dit.blocks,self.kv):block.cross._kv_cache=kv
        return self.dit(self.x,self.t,self.pos,ctx=self.ctx,lengths=self.lengths if self.masked else None,level=self.level)
torch.manual_seed(131);vm=VM().eval().requires_grad_(False);flow=RectifiedFlow();ctx=torch.randn(1,6,32)
engine=nf.Engine(vm,exact_graph=True,kvcache=True)
exact=ExactBackend(engine,flow,slot_cls=CPUStaticSlot,coarse="graph",warmup=0)
for seed in [0,42]:
    options=dict(K=8,select_level=3,seed=seed,thr=.4,cfg=[3.,3.,3.,5.],steps=20,sampler="dpmpp2m",max_nodes=64)
    with torch.inference_mode(),torch.autocast("cpu",dtype=torch.bfloat16):
        a,ai=sample_mbr(EagerBackend(vm,flow),vm,ctx,**options)
        b,bi=sample_mbr(exact,vm,ctx,**options)
    assert all(np.array_equal(x,y) for x,y in zip(a,b)) and ai==bi
print("CPU static-buffer/KV controls identical; CUDA graphs not exercised")
''')
