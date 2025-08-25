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

| Benchmark   | Paper                                     | GitHub                                         |
| ----------- | ----------------------------------------- | ---------------------------------------------- |
| LLaVA-Bench | [arXiv](https://arxiv.org/abs/2304.08485) | [GitHub](https://github.com/haotian-liu/LLaVA) |
| MM-IFEval   | [arXiv](https://arxiv.org/abs/2401.01314) | -                                              |
| MMDU        | [arXiv](https://arxiv.org/)               | -                                              |

#### Basic Ability

| Benchmark              | Paper                       | GitHub                                            |
| ---------------------- | --------------------------- | ------------------------------------------------- |
| OCRBench / OCRBench v2 | [arXiv](https://arxiv.org/) | [GitHub](https://github.com/BAAI-Agents/OCRBench) |
| ChartQA / ChartQAPro   | [arXiv](https://arxiv.org/) | -                                                 |
| MM-DocBench            | [arXiv](https://arxiv.org/) | -                                                 |

#### Comprehensive Perception

| Benchmark         | Paper                       | GitHub                                            |
| ----------------- | --------------------------- | ------------------------------------------------- |
| LVLM-eHub         | [arXiv](https://arxiv.org/) | [GitHub](https://github.com/OpenGVLab/LVLM-eHub)  |
| MMBench           | [arXiv](https://arxiv.org/) | [GitHub](https://github.com/open-compass/MMBench) |
| SEED-Bench Series | [arXiv](https://arxiv.org/) | -                                                 |

#### General Knowledge

| Benchmark | Paper                       | GitHub |
| --------- | --------------------------- | ------ |
| ScienceQA | [arXiv](https://arxiv.org/) | -      |
| EESE      | [arXiv](https://arxiv.org/) | -      |

#### Safety

| Benchmark      | Paper                       | GitHub |
| -------------- | --------------------------- | ------ |
| JailbreakV-28K | [arXiv](https://arxiv.org/) | -      |
| UnsafeBench    | [arXiv](https://arxiv.org/) | -      |
| SafeBench      | [arXiv](https://arxiv.org/) | -      |

### Specialized

#### Math

| Benchmark      | Paper                       | GitHub |
| -------------- | --------------------------- | ------ |
| MathVista      | [arXiv](https://arxiv.org/) | -      |
| PolyMATH       | [arXiv](https://arxiv.org/) | -      |
| Olympiad-Bench | [arXiv](https://arxiv.org/) | -      |

#### Physics

| Benchmark    | Paper                       | GitHub |
| ------------ | --------------------------- | ------ |
| MM-PhyQA     | [arXiv](https://arxiv.org/) | -      |
| PhysUniBench | [arXiv](https://arxiv.org/) | -      |
| PhysicsArena | [arXiv](https://arxiv.org/) | -      |

#### Chemistry

| Benchmark | Paper                       | GitHub |
| --------- | --------------------------- | ------ |
| ChemBench | [arXiv](https://arxiv.org/) | -      |
| ChemOCR   | [arXiv](https://arxiv.org/) | -      |

#### Finance

| Benchmark    | Paper                       | GitHub |
| ------------ | --------------------------- | ------ |
| FinMME       | [arXiv](https://arxiv.org/) | -      |
| Open-FinLLMs | [arXiv](https://arxiv.org/) | -      |

#### Healthcare & Medical Science

| Benchmark      | Paper                       | GitHub |
| -------------- | --------------------------- | ------ |
| HealthBench    | [arXiv](https://arxiv.org/) | -      |
| OpenMM-Medical | [arXiv](https://arxiv.org/) | -      |

#### Code

| Benchmark   | Paper                       | GitHub |
| ----------- | --------------------------- | ------ |
| Design2Code | [arXiv](https://arxiv.org/) | -      |
| HumanEval-V | [arXiv](https://arxiv.org/) | -      |

#### Earth Science / Remote Sensing

| Benchmark  | Paper                       | GitHub |
| ---------- | --------------------------- | ------ |
| GeoBench   | [arXiv](https://arxiv.org/) | -      |
| XLRS-Bench | [arXiv](https://arxiv.org/) | -      |

#### Embodied Tasks

| Benchmark     | Paper                       | GitHub |
| ------------- | --------------------------- | ------ |
| Ego4D         | [arXiv](https://arxiv.org/) | -      |
| EPIC-KITCHENS | [arXiv](https://arxiv.org/) | -      |

---

## Generation Evaluation

### Image

| Benchmark   | Paper                       | GitHub |
| ----------- | --------------------------- | ------ |
| Pick-a-Pic  | [arXiv](https://arxiv.org/) | -      |
| HPD v2      | [arXiv](https://arxiv.org/) | -      |
| ImageReward | [arXiv](https://arxiv.org/) | -      |

### Video

| Benchmark | Paper                       | GitHub |
| --------- | --------------------------- | ------ |
| Video-MME | [arXiv](https://arxiv.org/) | -      |
| MVBench   | [arXiv](https://arxiv.org/) | -      |

### Audio

| Benchmark  | Paper                       | GitHub |
| ---------- | --------------------------- | ------ |
| AudioBench | [arXiv](https://arxiv.org/) | -      |
| AIR-Bench  | [arXiv](https://arxiv.org/) | -      |

### 3D

| Benchmark     | Paper                       | GitHub |
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


We welcome pull requests (PRs)! If you contribute five or more valid benchmarks with relevant details, your contribution will be acknowledged in the next update of the paper's Acknowledgment section.
