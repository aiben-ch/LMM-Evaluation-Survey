# Large Multimodal Models Evaluation: A Survey

This repository complements the paper *Large Multimodal Models Evaluation: A Survey* and organizes benchmarks and resources across understanding (general and specialized), generation, and community platforms. It serves as a hub for researchers to find key datasets, papers, and code.

![Overview](overview.png)


**Paper:** [arXiv Preprint](https://arxiv.org/)
**Project Page:** [AIBench / LMM Evaluation Survey](https://github.com/aiben-ch/LMM-Evaluation-Survey)

---

## Table of Contents

1. [Understanding Evaluation](#understanding-evaluation)

   * [General](#general)

     * [Adaptability](#adaptability)
     * [Basic Ability](#basic-ability)
     * [Comprehensive Perception](#comprehensive-perception)
     * [General Knowledge](#general-knowledge)
     * [Safety](#safety)
   * [Specialized](#specialized)

     * [Math](#math)
     * [Physics](#physics)
     * [Chemistry](#chemistry)
     * [Finance](#finance)
     * [Healthcare & Medical Science](#healthcare--medical-science)
     * [Code](#code)
     * [Earth Science / Remote Sensing](#earth-science--remote-sensing)
     * [Embodied Tasks](#embodied-tasks)
2. [Generation Evaluation](#generation-evaluation)

   * [Image](#image)
   * [Video](#video)
   * [Audio](#audio)
   * [3D](#3d)
3. [Leaderboards and Tools](#leaderboards-and-tools)

---

## Understanding Evaluation

### General

#### Adaptability

| Benchmark   | Paper                                     | Project Page                                         |
| ----------- | ----------------------------------------- | ---------------------------------------------- |
| LLaVA-Bench | [Visual instruction tuning](https://arxiv.org/abs/2304.08485) | [GitHub](https://github.com/haotian-liu/LLaVA) |
| MIA-Bench   | [Mia-bench: Towards better instruction following evaluation of multimodal llms](https://arxiv.org/abs/2407.01509) | [Github](https://github.com/apple/ml-mia-bench) |
| MM-IFEval   | [Mm-ifengine: Towards multimodal instruction following](https://arxiv.org/abs/2401.01314) | -                                              |
| MMDU        | [arXiv](https://arxiv.org/)               | -                                              |

#### Basic Ability

| Benchmark              | Paper                       | Project Page                                            |
| ---------------------- | --------------------------- | ------------------------------------------------- |
| OCRBench / OCRBench v2 | [arXiv](https://arxiv.org/) | [GitHub](https://github.com/BAAI-Agents/OCRBench) |
| ChartQA / ChartQAPro   | [arXiv](https://arxiv.org/) | -                                                 |
| MM-DocBench            | [arXiv](https://arxiv.org/) | -                                                 |

#### Comprehensive Perception

| Benchmark         | Paper                       | Project Page                                            |
| ----------------- | --------------------------- | ------------------------------------------------- |
| LVLM-eHub         | [arXiv](https://arxiv.org/) | [GitHub](https://github.com/OpenGVLab/LVLM-eHub)  |
| MMBench           | [arXiv](https://arxiv.org/) | [GitHub](https://github.com/open-compass/MMBench) |
| SEED-Bench Series | [arXiv](https://arxiv.org/) | -                                                 |

#### General Knowledge

| Benchmark | Paper                       | Project Page |
| --------- | --------------------------- | ------ |
| ScienceQA | [arXiv](https://arxiv.org/) | -      |
| EESE      | [arXiv](https://arxiv.org/) | -      |

#### Safety

| Benchmark      | Paper                       | Project Page |
| -------------- | --------------------------- | ------ |
| JailbreakV-28K | [arXiv](https://arxiv.org/) | -      |
| UnsafeBench    | [arXiv](https://arxiv.org/) | -      |
| SafeBench      | [arXiv](https://arxiv.org/) | -      |

### Specialized

#### Math

| Benchmark      | Paper                       | Project Page |
| -------------- | --------------------------- | ------ |
| MathVista      | [arXiv](https://arxiv.org/) | -      |
| PolyMATH       | [arXiv](https://arxiv.org/) | -      |
| Olympiad-Bench | [arXiv](https://arxiv.org/) | -      |

#### Physics

| Benchmark    | Paper                       | Project Page |
| ------------ | --------------------------- | ------ |
| MM-PhyQA     | [arXiv](https://arxiv.org/) | -      |
| PhysUniBench | [arXiv](https://arxiv.org/) | -      |
| PhysicsArena | [arXiv](https://arxiv.org/) | -      |

#### Chemistry

| Benchmark | Paper                       | Project Page |
| --------- | --------------------------- | ------ |
| ChemBench | [arXiv](https://arxiv.org/) | -      |
| ChemOCR   | [arXiv](https://arxiv.org/) | -      |

#### Finance

| Benchmark    | Paper                       | Project Page |
| ------------ | --------------------------- | ------ |
| FinMME       | [arXiv](https://arxiv.org/) | -      |
| Open-FinLLMs | [arXiv](https://arxiv.org/) | -      |

#### Healthcare & Medical Science

| Benchmark      | Paper                       | Project Page |
| -------------- | --------------------------- | ------ |
| HealthBench    | [arXiv](https://arxiv.org/) | -      |
| OpenMM-Medical | [arXiv](https://arxiv.org/) | -      |

#### Code

| Benchmark   | Paper                       | Project Page |
| ----------- | --------------------------- | ------ |
| Design2Code | [arXiv](https://arxiv.org/) | -      |
| HumanEval-V | [arXiv](https://arxiv.org/) | -      |

#### Earth Science / Remote Sensing

| Benchmark  | Paper                       | Project Page |
| ---------- | --------------------------- | ------ |
| GeoBench   | [arXiv](https://arxiv.org/) | -      |
| XLRS-Bench | [arXiv](https://arxiv.org/) | -      |

#### Embodied Tasks

| Benchmark     | Paper                       | Project Page |
| ------------- | --------------------------- | ------ |
| Ego4D         | [arXiv](https://arxiv.org/) | -      |
| EPIC-KITCHENS | [arXiv](https://arxiv.org/) | -      |

---

## Generation Evaluation

### Image

| Benchmark   | Paper                       | Project Page |
| ----------- | --------------------------- | ------ |
| Pick-a-Pic  | [arXiv](https://arxiv.org/) | -      |
| HPD v2      | [arXiv](https://arxiv.org/) | -      |
| ImageReward | [arXiv](https://arxiv.org/) | -      |

### Video

| Benchmark | Paper                       | Project Page |
| --------- | --------------------------- | ------ |
| Video-MME | [arXiv](https://arxiv.org/) | -      |
| MVBench   | [arXiv](https://arxiv.org/) | -      |

### Audio

| Benchmark  | Paper                       | Project Page |
| ---------- | --------------------------- | ------ |
| AudioBench | [arXiv](https://arxiv.org/) | -      |
| AIR-Bench  | [arXiv](https://arxiv.org/) | -      |

### 3D

| Benchmark     | Paper                       | Project Page |
| ------------- | --------------------------- | ------ |
| M3DBench      | [arXiv](https://arxiv.org/) | -      |
| Space3D-Bench | [arXiv](https://arxiv.org/) | -      |

---

## Leaderboards and Tools

| Platform      | Link                                                    |
| ------------- | ------------------------------------------------------- |
| LMMs-Eval     | [GitHub](https://github.com/EvolvingLMMs-Lab/lmms-eval) |
| Chatbot Arena | [Website](https://lmsys.org/arena/)                     |
| OpenCompass   | [GitHub](https://github.com/open-compass/opencompass)   |

---
## Contributions

We welcome pull requests (PRs)! If you contribute five or more valid benchmarks with relevant details, your contribution will be acknowledged in the next update of the paper's Acknowledgment section.
