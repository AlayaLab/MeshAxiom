<div align="center">

# MeshAxiom: An Agent-Native 3D Foundation Stack for Programmable Worlds

[Yonghao Yu](https://scholar.google.com/citations?user=lwuwuAYAAAAJ) · [Ruihan Yu](https://auroraryan0301.github.io/) · [Sho Kuno](https://shokuno5.github.io/) · [Muyao Niu](https://myniuuu.github.io/) · [Yu-Ju Tsai](https://liagm.github.io/) · [Zheng-Hui Huang](https://brian90709.github.io/) · [Chuanhao Li](https://nevermorelch.github.io/) · [Fengbo Lan](https://openreview.net/profile?id=%7EFengbo_Lan1) · [Guixu Lin](https://www.researchgate.net/profile/Guixu_Lin) · [Runyi Li](https://lirunyi2001.github.io/) · [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/) · [Kaipeng Zhang](https://kpzhang93.github.io/)<sup>†</sup> · [Zhixiang Wang](https://lightchaserx.github.io/)<sup>†</sup>

<sup>†</sup> Corresponding authors


**[AlayaLab](https://github.com/AlayaLab)**

[![Code](https://img.shields.io/badge/Code-GitHub-24292F)](https://github.com/AlayaLab/MeshAxiom)
[![Project Page](https://img.shields.io/badge/Project-Page-1683D8)](https://alaya-lab.github.io/MeshAxiom/)
[![3D Asset Library](https://img.shields.io/badge/3D-Asset%20Library-5B6EE1)](https://alaya-lab.github.io/MeshAxiom/space-kit.html)

**Compact meshes. Explicit topology. Editable worlds.**

</div>

## Overview

MeshAxiom connects compact, explicit 3D meshes to persistent asset records and programmable scene state. **MeshAxiom-Base** turns a single image and a requested face budget into a native triangle mesh. File-based tools let programs and agents inspect and revise these assets, reuse them as modules, and control scene transforms, motion and cameras.

This is the official repository for the MeshAxiom model and workflow. The project website and interactive asset library are available now; code, model weights and the paper are planned for release here.

## Agent-Native 3D Authoring

Agent-native describes the interface: agents call tools on asset files, receive compact summaries and rendered views, and retain geometry and generation recipes for later operations. Compact geometry and explicit triangle connectivity provide objects that geometric code and scene programs can read and edit.

MeshAxiom-Base generates vertices coarse to fine with a diffusion transformer over a sparse octree, then predicts connectivity with a second diffusion transformer in the latent space of a topology autoencoder. The requested face budget conditions generation; realized counts can differ from the request.

## From Meshes to Worlds

The authoring workflow carries generated geometry into ordinary 3D tools:

1. **Generate meshes:** create compact geometry with explicit vertices and triangle faces.
2. **Assemble scenes:** arrange assets and parts in an editable 3D scene.
3. **Author materials and motion:** add UVs, materials, rigs, part transforms and animation.
4. **Render the scene:** define cameras, lighting and effects to produce geometry and material control videos.

A video-generation model can use a control video as a soft condition for an additional appearance stage. The base model generates mesh geometry and connectivity; UVs, materials, rigs, scene assembly and choreography belong to the authoring workflow.

## Explore Space Saga

**[Space Saga](https://alaya-lab.github.io/MeshAxiom/#saga)** showcases twelve scenes built from the space asset kit, including stations, fleets, shipyards, moon bases and alien encounters. Compare **Materials** with **Gray mesh** or **Topology** on the same scene timeline.

The twelve scenes use generated meshes in explicit scene programs. Bounded mesh repairs and external material refinement belong to the authoring workflow, separate from raw-model measurements.

The **[3D asset library](https://alaya-lab.github.io/MeshAxiom/space-kit.html)** contains **123 assets**, with input-image, gray-mesh, topology and material views, interactive 3D inspection and original GLB downloads. Explore modular assemblies and scenes built from the kit, with **Materials**, **Gray mesh** and **Topology** comparisons.

## Release Status

- [x] Project website and Space Saga showcase
- [x] Interactive 3D asset library and asset downloads
- [ ] MeshAxiom-Base code
- [ ] Model weights
- [ ] MeshAxiom workflow and tools
- [ ] Paper and workflow documentation

Installation instructions, model download links and paper details will be added here as each release becomes available.
