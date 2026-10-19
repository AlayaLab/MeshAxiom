# MeshAxiom

**面向 Agent 的三维基础模型与可编程世界工作流**

[项目主页](https://alaya-lab.github.io/MeshAxiom/) · [Hugging Face](https://huggingface.co/AlayaLab/MeshAxiom-Base) · [GitHub](https://github.com/AlayaLab/MeshAxiom)

本包只提供 **Exact 无损加速推理**。当前 Base 配方为顶点 EMA 100K、拓扑 EMA 10K、VAE EMA 30K，使用 MBR8、20 步和拓扑 CFG 3，接入 Sho Kuno 的 CUDA Graph 与 K/V 缓存实现。模型常驻两个隔离 worker，连续请求复用模型。

## 安装与生成

作者加速环境为 Linux、H200、NVIDIA PyTorch 25.03、CUDA 12.8、Python 3.12。建议使用独立环境；其他硬件和软件组合先做一致性验证。

```bash
python -m pip install -e '.[geometry,hub,background,postprocess]'
meshaxiom doctor --config models/MeshAxiom-Base/meshaxiom.json --operation geometry --verify-hashes --runtime-check
meshaxiom generate --config models/MeshAxiom-Base/meshaxiom.json --image image.png --face-budget 4000 --output outputs/object
meshaxiom serve --config models/MeshAxiom-Base/meshaxiom.json --workspace outputs/server
```

浏览器打开 `http://127.0.0.1:7860` 上传图片。透明 PNG 可使用 `--background alpha`；已经处理好的 512×512 黑底 RGB 图片使用 `--background keep`；普通图片默认 `auto` 去背景。

```python
from meshaxiom import MeshAxiomPipeline

with MeshAxiomPipeline.from_pretrained("models/MeshAxiom-Base") as pipeline:
    result = pipeline.generate("image.png", "outputs/object", face_budget=4000)
```

[MeshAxiom-Base 模型库](https://huggingface.co/AlayaLab/MeshAxiom-Base)目前需要授权访问，可用模型 ID `AlayaLab/MeshAxiom-Base` 和已验收的完整 commit SHA `25a6250fc88e2438e5a666ad8622c3475f64e6b0` 加载。GitHub 只放代码；模型目录含三份纯 EMA 权重和 DINOv3 配置。冻结的 DINOv3 张量已包含在 EMA 中，无需再次下载一整份编码器权重。

面数请求是生成条件，实际面数如实报告，不强制减面。输出包括 OBJ、GLB、JSON、顶点和原始拓扑，以及输入和权重校验记录。首次请求需要加载模型；服务中的后续请求直接复用。使用 `with` 或 `close()` 释放模型。

## 无损验证

```bash
meshaxiom verify-exact --config models/MeshAxiom-Base/meshaxiom.json --image examples/reference.png --background keep --face-budget 4000 --output outputs/identity-4k
```

当前版本 `0.2.0.dev2` 已通过 95 项 CPU 测试，并在 H200 上完成同一张图片的 4K、8K 请求验证：四份几何文件逐字节一致，连续请求复用同一组 worker。结果仅覆盖该图片、种子和软件环境，详见 [验收记录](VALIDATION.json)。

检查四份几何文件的完整 SHA256。一致性不通过会返回失败并保留结果，不能把“生成成功”当作无损验证通过。此命令中的原始实现仅用于验证。

Agent 使用 [JSON 工具与 HTTP 接口](docs/AGENT_USAGE.md)。[材质生成](docs/MATERIALS.md) 独立提供，不需要先加载几何模型。Hugging Face 模型与 Space 的准备方法见 [部署说明](docs/HUGGINGFACE.md)。

本目录为代码发布候选；模型库目前需要授权访问，托管 Space 尚未上线。许可证、真实 Hub revision 和目标机器验收状态见 [发布状态](docs/RELEASE_PREPARATION.md)。
