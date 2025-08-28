# Large Multimodal Models Evaluation: A Survey

This repository complements the paper *Large Multimodal Models Evaluation: A Survey* and organizes benchmarks and resources across understanding (general and specialized), generation, and community platforms. It serves as a hub for researchers to find key datasets, papers, and code.

![Overview](overview.png)


**Paper:** [Preprint](https://github.com/aiben-ch/LMM-Evaluation-Survey/blob/main/Large_Multimodal_Models_Evaluation__A_Survey_Preprint.pdf)
**Project Page:** [AIBench / LMM Evaluation Survey](https://github.com/aiben-ch/LMM-Evaluation-Survey)

---
## Contributions

We welcome pull requests (PRs)! If you contribute five or more valid benchmarks with relevant details, your contribution will be acknowledged in the next update of the paper's Acknowledgment section.

Come on and join us !!

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

|          Benchmark          |                            Paper                             |                         Project Page                         |
| :-------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|         LLaVA-Bench         | [Visual instruction tuning](https://arxiv.org/abs/2304.08485) |        [Github](https://github.com/haotian-liu/LLaVA)        |
|          MIA-Bench          | [Mia-bench: Towards better instruction following evaluation of multimodal llms](https://arxiv.org/abs/2407.01509) |       [Github](https://github.com/apple/ml-mia-bench)        |
|          MM-IFEval          | [MM-IFEngine: Towards Multimodal Instruction Following](https://arxiv.org/abs/2504.07957) |       [Github](https://github.com/SYuan03/MM-IFEngine)       |
|         VisIT-Bench         | [VisIT-Bench: A Benchmark for Vision-Language Instruction Following Inspired by Real-World Use](https://arxiv.org/pdf/2308.06595) |   [Github](https://github.com/mlfoundations/VisIT-Bench/)    |
|            MMDU             | [MMDU: A Multi-Turn Multi-Image Dialog Understanding Benchmark and Instruction-Tuning Dataset for LVLMs](https://arxiv.org/abs/2406.11833) |         [Github](https://github.com/Liuziyu77/MMDU)          |
|          ConvBench          | [ConvBench: A Multi-Turn Conversation  Evaluation Benchmark with Hierarchical  Capability for Large Vision-Language Models](https://arxiv.org/pdf/2403.20194) |      [Github](https://github.com/shirlyliu64/ConvBench)      |
|          SIMMC 2.0          | [SIMMC 2.0: A Task-oriented Dialog Dataset for Immersive Multimodal Conversations](https://arxiv.org/pdf/2104.08667) |     [Github](https://github.com/facebookresearch/simmc2)     |
|          Mementos           | [Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences](https://arxiv.org/pdf/2401.10529) |     [Github](https://github.com/umd-huang-lab/Mementos)      |
|          MUIRBENCH          | [MUIRBENCH: A Comprehensive Benchmark for Robust Multi-image Understanding](https://arxiv.org/pdf/2406.09411) |       [Github](https://github.com/muirbench/MuirBench)       |
|            MMIU             | [MMIU: Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models](https://arxiv.org/pdf/2408.02718) |         [Github](https://github.com/OpenGVLab/MMIU)          |
|            MIRB             | [Benchmarking Multi-Image Understanding in Vision and Language Models: Perception, Knowledge, Reasoning, and Multi-Hop Reasoning ](http://arxiv.org/pdf/2406.12742) |          [Github](https://github.com/ys-zong/MIRB)           |
|           MIBench           | [MIBench: Evaluating Multimodal Large Language Models over Multiple Images](https://arxiv.org/abs/2407.15272) | [Hugging Face](https://huggingface.co/datasets/StarBottle/MIBench) |
|          II-Bench           | [II-Bench: An Image Implication Understanding Benchmark for Multimodal Large Language Models](https://arxiv.org/abs/2406.05862) |        [Github](https://github.com/II-Bench/II-Bench)        |
|           Mantis            | [Mantis: Interleaved Multi-Image Instruction Tuning](https://arxiv.org/pdf/2405.01483) |       [Github](https://github.com/TIGER-AI-Lab/Mantis)       |
|          MileBench          | [MILEBENCH: Benchmarking MLLMs in Long Context](https://arxiv.org/pdf/2404.18532) |       [Github](https://github.com/MileBench/MileBench)       |
|            ReMI             | [ReMI: A Dataset for Reasoning with Multiple Images](https://arxiv.org/pdf/2406.09175) | [Hugging Face](https://huggingface.co/datasets/mehrankazemi/ReMI) |
|            CODIS            | [CODIS: Benchmarking Context-Dependent Visual Comprehension for Multimodal Large Language Models](https://arxiv.org/pdf/2402.13607) |         [Github](https://github.com/THUNLP-MT/CODIS)         |
|          SPARKLES           | [SPARKLES: UNLOCKING CHATS ACROSS MULTIPLE IMAGES FOR MULTIMODAL INSTRUCTION-FOLLOWING MODELS ](https://arxiv.org/pdf/2308.16463) |        [Github](https://github.com/HYPJUDY/Sparkles)         |
|            MMIE             | [MMIE: MASSIVE MULTIMODAL INTERLEAVED COMPREHENSION BENCHMARK FOR LARGE VISIONLANGUAGE MODELS](https://arxiv.org/pdf/2410.10139) |        [Github](https://github.com/Lillianwei-h/MMIE)        |
|      InterleavedBench       | [Holistic Evaluation for Interleaved Text-and-Image Generation](https://arxiv.org/pdf/2406.14643) | [Hugging Face](https://huggingface.co/mqliu/InterleavedBench) |
|           OpenING           | [OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation](https://arxiv.org/pdf/2411.18499) |        [Github](https://github.com/LanceZPF/OpenING)         |
|         HumaniBench         | [HumaniBench: A Human-Centric Framework for Large Multimodal Models Evaluation](https://arxiv.org/pdf/2505.11454) |   [Github](https://github.com/VectorInstitute/HumaniBench)   |
|         Herm-Bench          | [HERM: Benchmarking and Enhancing Multimodal LLMs for Human-Centric Understanding](https://arxiv.org/pdf/2410.06777) |  [Github](https://github.com/ZJHTerry18/Human-Centric-MLLM)  |
|            UNIAA            | [UNIAA: A Unified Multi-modal Image Aesthetic Assessment Baseline and Benchmark](https://arxiv.org/pdf/2404.09619) |          [Github](https://github.com/KwaiVGI/Uniaa)          |
|         Humanbeauty         | [HumanAesExpert: Advancing a Multi-Modality Foundation Model for Human Image Aesthetic Assessment](https://arxiv.org/pdf/2503.23907) |     [Github](https://github.com/KwaiVGI/HumanAesExpert)      |
|          SocialIQA          | [SOCIAL IQA: Commonsense Reasoning about Social Interactions](https://arxiv.org/pdf/1904.09728) | [Hugging Face](https://huggingface.co/datasets/allenai/social_i_qa) |
|       EmpathicStories       | [EmpathicStories++: A Multimodal Dataset for Empathy towards Personal Experiences](https://arxiv.org/pdf/2405.15708) | [Dataset Download](https://mitmedialab.github.io/empathic-stories-multimodal/) |
|        Chatbot Arena        | [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/pdf/2403.04132) |                       *Not available*                        |
| OpenAssistant Conversations | [OpenAssistant Conversations - Democratizing Large Language Model Alignment ](https://arxiv.org/pdf/2304.07327) |     [Github](https://github.com/LAION-AI/Open-Assistant)     |
|             HCE             | [Human-Centric Evaluation for Foundation Models ](https://arxiv.org/pdf/2506.01793) | [Github](https://github.com/yijinguo/Human-Centric-Evaluation) |

#### Basic Ability

|     Benchmark     |                            Paper                             |                         Project Page                         |
| :---------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|     NWPU-MOC      | [ NWPU-MOC: A Benchmark for Fine-grained  Multi-category Object Counting in Aerial Images](https://arxiv.org/pdf/2401.10530) |         [Github](https://github.com/lyongo/NWPU-MOC)         |
|   T2V-ComBench    | [T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation](https://arxiv.org/pdf/2407.14505) | [Github](https://github.com/KaiyueSun98/T2V-CompBench/tree/V2) |
|    ConceptMix     | [ConceptMix: A Compositional Image Generation Benchmark with Controllable Difficulty](https://arxiv.org/pdf/2408.14339) |  [Github](https://github.com/princetonvisualai/ConceptMix)   |
|       PICD        | [Can Machines Understand Composition? Dataset and Benchmark for Photographic Image Composition Embedding and Understanding](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_Can_Machines_Understand_Composition_Dataset_and_Benchmark_for_Photographic_Image_CVPR_2025_paper.pdf) | [Github](https://github.com/CV-xueba/PICD_ImageComposition)  |
|      TextVQA      | [Towards VQA Models That Can Read](https://arxiv.org/pdf/1904.08920) |      [Github](https://github.com/facebookresearch/mmf)       |
|      OCR-VQA      | [OCR-VQA: Visual Question Answering by Reading Text in Images](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8978122) |        [Dataset download](https://ocr-vqa.github.io/)        |
|     OCRBench      | [OCRBENCH: ON THE HIDDEN MYSTERY OF OCR IN LARGE MULTIMODAL MODELS](https://arxiv.org/pdf/2305.07895) |    [Github](https://github.com/Yuliang-Liu/MultimodalOCR)    |
|    OCRBench v2    | [OCRBench v2: An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning](https://arxiv.org/pdf/2501.00321) |    [Github](https://github.com/Yuliang-Liu/MultimodalOCR)    |
|     ASCIIEval     | [VISUAL PERCEPTION IN TEXT STRINGS](https://arxiv.org/pdf/2410.01733) |     [Github](https://github.com/JiaQiSJTU/VisionInText)      |
|   OCRReasoning    | [OCR-Reasoning Benchmark: Unveiling the True Capabilities of MLLMs in Complex Text-Rich Image Reasoning ](https://arxiv.org/pdf/2505.17163) |   [Github](https://github.com/SCUT-DLVCLab/OCR-Reasoning)    |
|    M4-ViteVQA     | [Towards Video Text Visual Question Answering: Benchmark and Baseline](https://proceedings.neurips.cc/paper_files/paper/2022/file/e726197ffd401df4013cd9f81007b5cf-Paper-Datasets_and_Benchmarks.pdf) |         [Github](https://github.com/bytedance/VTVQA)         |
| SEED-Bench-2-Plus | [SEED-Bench-2-Plus: Benchmarking Multimodal Large Language Models with Text-Rich Visual Comprehension](https://arxiv.org/pdf/2404.16790) |      [Github](https://github.com/AILab-CVC/SEED-Bench)       |
|    MMDocBench     | [MMDOCBENCH: BENCHMARKING LARGE VISIONLANGUAGE MODELS FOR FINE-GRAINED VISUAL DOCUMENT UNDERSTANDING](https://arxiv.org/pdf/2410.21311) |      [Github](https://github.com/fengbinzhu/MMDocBench)      |
|  MMLongBench-Doc  | [MMLONGBENCH-DOC: Benchmarking Long-context Document Understanding with Visualizations ](https://arxiv.org/pdf/2407.01523) |   [Github](https://github.com/mayubo2333/MMLongBench-Doc)    |
|        UDA        | [UDA: A Benchmark Suite for Retrieval Augmented Generation in Real-world Document Analysis](https://arxiv.org/pdf/2406.15187) |    [Github](https://github.com/qinchuanhui/UDA-Benchmark)    |
|     VisualMRC     | [VisualMRC: Machine Reading Comprehension on Document Images ](https://cdn.aaai.org/ojs/17635/17635-13-21129-1-2-20210518.pdf) |     [Github](https://github.com/nttmdlab-nlp/VisualMRC)      |
|      DocVQA       | [DocVQA: A Dataset for VQA on Document Images](https://arxiv.org/pdf/2007.00398) | [Hugging Face](https://huggingface.co/datasets/eliolio/docvqa) |
|     DocGenome     | [DocGenome: An Open Large-scale Scientific Document Benchmark for Training and Testing Multi-modal Large Language Models](https://arxiv.org/pdf/2406.11633) |    [Github](https://github.com/Alpha-Innovator/DocGenome)    |
|     GDI-Bench     | [GDI-Bench: A Benchmark for General Document Intelligence with Vision and Reasoning Decoupling](https://arxiv.org/pdf/2505.00063) | [Dataset download](https://knowledgexlab.github.io/gdibench.github.io/) |
|       AitW        | [Android in the Wild: A Large-Scale Dataset for Android Device Control](https://arxiv.org/pdf/2307.10088) | [Github](https://github.com/google-research/google-research/tree/master/android_in_the_wild) |
|    ScreenSpot     | [SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents](https://arxiv.org/pdf/2401.10935) |       [Github](https://github.com/njucckevin/SeeClick)       |
|  VisualWebBench   | [VisualWebBench: How Far Have Multimodal LLMs Evolved in Web Page Understanding and Grounding?](https://arxiv.org/pdf/2404.05955) |  [Github](https://github.com/VisualWebBench/VisualWebBench)  |
|     GUI-WORLD     | [GUI-WORLD: A VIDEO BENCHMARK AND DATASET FOR MULTIMODAL GUI-ORIENTED UNDERSTANDING](https://arxiv.org/pdf/2406.10819) |     [Github](https://github.com/Dongping-Chen/GUI-World)     |
|    WebUIBench     | [WebUIBench: A Comprehensive Benchmark for Evaluating Multimodal Large Language Models in WebUI-to-Code ](https://www.arxiv.org/pdf/2506.07818) |     [Github](https://github.com/MAIL-Tele-AI/WebUIBench)     |
|     ScreenQA      | [ScreenQA: Large-Scale Question-Answer Pairs Over Mobile App Screenshots ](https://arxiv.org/pdf/2209.08199) | [Github](https://github.com/google-research-datasets/screen_qa) |
|      ChartQA      | [ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning](https://arxiv.org/pdf/2203.10244) |         [Github](https://github.com/vis-nlp/ChartQA)         |
|    ChartQApro     | [CHARTQAPRO : A More Diverse and Challenging Benchmark for Chart Question Answering](https://arxiv.org/pdf/2504.05506) |       [Github](https://github.com/vis-nlp/ChartQAPro)        |
|      ComTQA       | [TabPedia: Towards Comprehensive Visual Table Understanding with Concept Synergy](https://arxiv.org/pdf/2406.01326) |    [Github](https://github.com/sakura2233565548/TabPedia)    |
|  TableVQA-Bench   | [TableVQA-Bench: A Visual Question Answering Benchmark on Multiple Table Domains ](https://arxiv.org/pdf/2404.19205) |     [Github](https://github.com/naver-ai/tablevqabench)      |
|      CharXiv      | [CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs](https://arxiv.org/pdf/2406.18521) |      [Github](https://github.com/princeton-nlp/CharXiv)      |
|    SciFIBench     | [SciFIBench: Benchmarking Large Multimodal Models for Scientific Figure Interpretation](https://arxiv.org/pdf/2405.08807) |  [Github](https://github.com/jonathan-roberts1/SciFIBench)   |
|     AI2D-RST      | [AI2D-RST: A multimodal corpus of 1000 primary school science diagrams](https://arxiv.org/pdf/1912.03879) |        [Github](https://github.com/thiippal/AI2D-RST)        |
|    InfoChartQA    | [InfoChartQA: A Benchmark for Multimodal Question Answering on Infographic Charts](https://arxiv.org/pdf/2505.19028) |     [Github](https://github.com/CoolDawnAnt/InfoChartQA)     |
|    EvoChart-QA    | [EvoChart: A Benchmark and a Self-Training Approach Towards Real-World Chart Understanding](https://arxiv.org/pdf/2409.01577) |       [Github](https://github.com/MuyeHuang/EvoChart)        |
|     WikiMixQA     | [WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts](https://arxiv.org/pdf/2506.15594) |    [Github](https://github.com/negar-foroutan/WikiMixQA)     |
|      ChartX       | [ChartX & ChartVLM: A Versatile Benchmark and Foundation Model for Complicated Chart Reasoning](https://arxiv.org/pdf/2402.12185) |    [Github](https://github.com/Alpha-Innovator/ChartVLM)     |
|      Q-Bench      | [Q-BENCH: A BENCHMARK FOR GENERAL-PURPOSE FOUNDATION MODELS ON LOW-LEVEL VISION ](https://arxiv.org/pdf/2309.14181) |        [Github](https://github.com/Q-Future/Q-Bench)         |
|      A-Bench      | [A-BENCH: ARE LMMS MASTERS AT EVALUATING AI-GENERATED IMAGES?](https://arxiv.org/pdf/2406.03070) |        [Github](https://github.com/Q-Future/A-Bench)         |
|     MVP-Bench     | [MVP-Bench: Can Large Vision–Language Models Conduct Multi-level Visual Perception Like Humans?](https://arxiv.org/pdf/2410.04345) |      [Github](https://github.com/GuanzhenLi/MVP-Bench)       |
|    XLRS-Bench     | [XLRS-Bench: Could Your Multimodal LLMs Understand Extremely Large Ultra-High-Resolution Remote Sensing Imagery?](https://arxiv.org/pdf/2503.23771) |       [Github](https://github.com/AI9Stars/XLRS-Bench)       |
|     HR-Bench      | [Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models ](https://arxiv.org/pdf/2408.15556) |        [Github](https://github.com/DreamMr/HR-Bench)         |
|   MME-RealWorld   | [MME-RealWorld: Could Your Multimodal LLM Challenge High-Resolution Real-World Scenarios that are Difficult for Humans?](https://arxiv.org/pdf/2408.13257) |  [Github](https://github.com/MME-Benchmarks/MME-RealWorld)   |
|      V*Bench      | [V ∗ : Guided Visual Search as a Core Mechanism in Multimodal LLMs](https://arxiv.org/pdf/2312.14135) |        [Github](https://github.com/penghao-wu/vstar)         |
|     FaceBench     | [FaceBench: A Multi-View Multi-Level Facial Attribute VQA Dataset for Benchmarking Face Perception MLLMs](https://arxiv.org/pdf/2503.21457) |        [Github](https://github.com/CVI-SZU/FaceBench)        |
|    MMAFFBench     | [MMAFFBen: A Multilingual and Multimodal Affective Analysis Benchmark for Evaluating LLMs and VLMs](https://arxiv.org/pdf/2505.24423) |         [Github](https://github.com/lzw108/MMAFFBen)         |
|    FABA-Bench     | [Facial Affective Behavior Analysis with Instruction Tuning ](https://arxiv.org/pdf/2404.05052) |          [Github](https://github.com/JackYFL/EmoLA)          |
|    MEMO-Bench     | [MEMO-Bench: A Multiple Benchmark for Text-to-Image and Multimodal Large Language Models on Human Emotion Analysis ](https://arxiv.org/pdf/2411.11235) |       [Github](https://github.com/zyj-2000/MEMO-Bench)       |
|     EmoBench      | [EmoBench: Evaluating the Emotional Intelligence of Large Language Models](https://arxiv.org/pdf/2402.12071) |       [Github](https://github.com/Sahandfer/EmoBench)        |
|    EEmo-Bench     | [EEmo-Bench: A Benchmark for Multi-modal Large Language Models on Image Evoked Emotion Assessment ](https://arxiv.org/pdf/2504.16405) |      [Github](https://github.com/workerred/EEmo-Bench)       |
|     AesBench      | [AesBench: An Expert Benchmark for Multimodal Large Language Models on Image Aesthetics Perception](https://arxiv.org/pdf/2401.08276) |         [Github](https://github.com/yipoh/AesBench)          |
|    UNIAA-Bench    | [UNIAA: A Unified Multi-modal Image Aesthetic Assessment Baseline and Benchmark](https://arxiv.org/pdf/2404.09619) |          [Github](https://github.com/KwaiVGI/Uniaa)          |
|    ImplictAVE     | [ImplicitAVE: An Open-Source Dataset and Multimodal LLMs Benchmark for Implicit Attribute Value Extraction](https://arxiv.org/pdf/2404.15592) |    [Github](https://github.com/HenryPengZou/ImplicitAVE)     |
|     II-Bench      | [II-Bench: An Image Implication Understanding Benchmark for Multimodal Large Language Models](https://arxiv.org/pdf/2406.05862) | [Hugging Face](https://huggingface.co/datasets/m-a-p/II-Bench) |
|     CogBench      | [A Cognitive Evaluation Benchmark of Image Reasoning and Description for Large Vision-Language Models](https://arxiv.org/pdf/2402.18409) |        [Github](https://github.com/X-LANCE/CogBench)         |
|      A4Bench      | [Affordance Benchmark for MLLMs](https://arxiv.org/pdf/2506.00893) |     [Github](https://github.com/JunyingWang959/A4Bench)      |
|      MM-SAP       | [MM-SAP: A Comprehensive Benchmark for Assessing Self-Awareness of Multimodal Large Language Models in Perception](https://arxiv.org/pdf/2401.07529) |          [Github](https://github.com/YHWmz/MM-SAP)           |
|    Cambrian-1     | [Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs](https://arxiv.org/pdf/2406.16860) |     [Github](https://github.com/cambrian-mllm/cambrian)      |
|     MMUBench      | [Single Image Unlearning: Efficient Machine Unlearning in Multimodal Large Language Models](https://arxiv.org/pdf/2405.12523) |                       *Not available*                        |
|       MMVP        | [Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs](https://arxiv.org/pdf/2401.06209) |          [Github](https://github.com/tsb0601/MMVP)           |
|     MicBench      | [Towards Open-ended Visual Quality Comparison](https://arxiv.org/pdf/2402.16641) |      [Github](https://github.com/Q-Future/Co-Instruct)       |
|    CuturalVQA     | [Benchmarking Vision Language Models for Cultural Understanding](https://arxiv.org/pdf/2407.10920) | [Hugging Face](https://huggingface.co/datasets/mair-lab/CulturalVQA) |
|  RefCOCO Family   | [Modeling Context in Referring Expressions](https://arxiv.org/pdf/1511.02283)[Generation and Comprehension of Unambiguous Object Descriptions](https://arxiv.org/pdf/1511.02283) | [Github](https://github.com/lichengunc/refer) [Github](https://github.com/mjhucla/Google_Refexp_toolbox) |
|      Ref-L4       | [Revisiting Referring Expression Comprehension Evaluation in the Era of Large Multimodal Models](https://arxiv.org/pdf/2406.16866) |        [Github](https://github.com/JierunChen/Ref-L4)        |
|     MRES-32M      | [Unveiling Parts Beyond Objects: Towards Finer-Granularity Referring Expression Segmentation ](https://arxiv.org/pdf/2312.08007) |        [Github](https://github.com/Rubics-Xuan/MRES)         |
|      UrBench      | [UrBench: A Comprehensive Benchmark for Evaluating Large Multimodal Models in Multi-View Urban Scenarios ](https://arxiv.org/pdf/2408.17267) |       [Github](https://github.com/opendatalab/UrBench)       |
|      COUNTS       | [COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models under Distribution Shifts](https://arxiv.org/pdf/2504.10158) |  [Github](https://github.com/jiansheng-li/COUNTS_benchmark)  |
|       MTVQA       | [MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering](https://arxiv.org/pdf/2405.11985) |         [Github](https://github.com/bytedance/MTVQA)         |
|     GePBench      | [GePBench: Evaluating Fundamental Geometric Perception for Multimodal Large Language Models ](https://arxiv.org/pdf/2412.21036) |                       *Not available*                        |
|    SpatialMQA     | [Can Multimodal Large Language Models Understand Spatial Relations?](https://arxiv.org/pdf/2505.19015) |     [Github](https://github.com/ziyan-xiaoyu/SpatialMQA)     |
| SpacialRGPT-Bench | [SpatialRGPT: Grounded Spatial Reasoning in Vision-Language Models](https://arxiv.org/pdf/2406.01584) |     [Github](https://github.com/AnjieCheng/SpatialRGPT)      |
|      CoSpace      | [CoSpace: Benchmarking Continuous Space Perception Ability for Vision-Language Models ](https://arxiv.org/pdf/2503.14161) |       [Github](https://github.com/THUNLP-MT/CoSpace/)        |
|   LMM-CompBench   | [MLLM-COMPBENCH: A Comparative Reasoning Benchmark for Multimodal LLMs](https://arxiv.org/pdf/2407.16837) |    [Github](https://github.com/RaptorMai/MLLM-CompBench)     |
|     SOK-Bench     | [SOK-Bench: A Situated Video Reasoning Benchmark with Aligned Open-World Knowledge](https://arxiv.org/pdf/2405.09713) |        [Github](https://github.com/csbobby/SOK-Bench)        |
|     GSR-Bench     | [GSR-BENCH: A Benchmark for Grounded Spatial Reasoning Evaluation via Multimodal LLMs](https://arxiv.org/pdf/2406.13246) |                       *Not available*                        |
|    What's "up"    | [What’s “up” with vision-language models? Investigating their struggle with spatial reasoning](https://arxiv.org/pdf/2310.19785) |    [Github](https://github.com/amitakamath/whatsup_vlms)     |
|  Q-Spatial Bench  | [Reasoning Paths with Reference Objects Elicit Quantitative Spatial Reasoning in Large Vision-Language Models](https://arxiv.org/pdf/2409.09788) | [Github](https://github.com/andrewliao11/Q-Spatial-Bench-code) |
|       AS-V2       | [The All-Seeing Project V2: Towards General Relation Comprehension of the Open World](https://arxiv.org/pdf/2402.19474) |      [Github](https://github.com/OpenGVLab/all-seeing)       |
|    Visual CoT     | [Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning](https://arxiv.org/pdf/2403.16999) |      [Github](https://github.com/deepcs233/Visual-CoT)       |
|    LogicVista     | [LogicVista: Multimodal LLM Logical Reasoning Benchmark in Visual Contexts ](https://arxiv.org/pdf/2407.04973) |      [Github](https://github.com/Yijia-Xiao/LogicVista)      |
|     VisuLogic     | [VisuLogic: A Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models](https://arxiv.org/pdf/2504.15279) |       [Github](https://github.com/VisuLogic-Benchmark)       |
|       CoMT        | [CoMT: A Novel Benchmark for Chain of Multi-modal Thought on Large Vision-Language Models](https://arxiv.org/pdf/2412.12932) |           [Github](https://github.com/czhhzc/CoMT)           |
|      PUZZLES      | [PUZZLES: A Benchmark for Neural Algorithmic Reasoning ](https://arxiv.org/pdf/2407.00401) |          [Github](https://github.com/ETH-DISCO/rlp)          |
|       LOVA3       | [LOVA3 : Learning to Visual Question Answering, Asking and Assessment](https://arxiv.org/pdf/2405.14974) |          [Github](https://github.com/showlab/LOVA3)          |
|      VLIKEB       | [VLKEB: A Large Vision-Language Model Knowledge Editing Benchmark](https://arxiv.org/pdf/2403.07350) |           [Github](https://github.com/VLKEB/VLKEB)           |
|    MMKE-Bench     | [MMKE-BENCH: A MULTIMODAL EDITING BENCHMARK FOR DIVERSE VISUAL KNOWLEDGE](https://arxiv.org/pdf/2502.19870) |   [Github](https://github.com/MMKE-Bench-ICLR/MMKE-Bench)    |
|      MC-MKE       | [MC-MKE: A Fine-Grained Multimodal Knowledge Editing Benchmark Emphasizing Modality Consistency](https://arxiv.org/pdf/2406.13219)[MIKE: A New Benchmark for Fine-grained Multimodal Entity Knowledge Editing](https://arxiv.org/pdf/2402.14835) |                       *Not available*                        |
|      NegVQA       | [NegVQA: Can Vision Language Models Understand Negation? ](https://arxiv.org/pdf/2505.22946) |        [Github](https://github.com/yuhui-zh15/NegVQA)        |

#### Comprehensive Perception

|     Benchmark     |                            Paper                             |                         Project Page                         |
| :---------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|     LVLM-eHub     | [Lvlm-ehub: A comprehensive evaluation benchmark for large vision-language models.](https://arxiv.org/abs/2306.09265) | [GitHub](https://github.com/OpenGVLab/Multi-Modality-Arena)  |
|   TinyLVLM-eHub   | [Tinylvlm-ehub: Towards comprehensive and efficient evaluation for large vision-language models.](https://arxiv.org/abs/2308.03729) | [GitHub](https://github.com/OpenGVLab/Multi-Modality-Arena)  |
|       LAMM        | [Lamm: Language-assisted multi-modal instruction-tuning dataset, framework, and benchmark.](https://arxiv.org/abs/2306.06687) |          [GitHub](https://github.com/OpenLAMM/LAMM)          |
|        MME        | [Mme: A comprehensive evaluation benchmark for multimodal large language models.](https://arxiv.org/abs/2306.13394) | [Project Page](https://mme-benchmark.github.io/home_page.html) |
|      MMBench      | [Mmbench: Is your multi-modal model an all-around player?](https://arxiv.org/abs/2307.06281) |      [GitHub](https://github.com/open-compass/mmbench)       |
| SEED-Bench series | [SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension.](https://arxiv.org/abs/2307.16125) |      [GitHub](https://github.com/AILab-CVC/SEED-Bench)       |
|     MMT-Bench     | [Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi.](https://arxiv.org/abs/2404.16006) |       [GitHub](https://github.com/OpenGVLab/MMT-Bench)       |
|     LMMs-Eval     | [Lmms-eval: Reality check on the evaluation of large multimodal models.](https://arxiv.org/abs/2407.12772) |   [GitHub](https://github.com/EvolvingLMMs-Lab/lmms-eval)    |
|      MMStar       | [Are we on the right way for evaluating large vision-language models?](https://arxiv.org/abs/2403.20330) |     [GitHub](https://github.com/MMStar-Benchmark/MMStar)     |
|   NaturalBench    | [Naturalbench: Evaluating vision-language models on natural adversarial samples.](https://arxiv.org/abs/2410.14669) |   [Project Page](https://github.com/Baiqi-Li/NaturalBench)   |
|      MM-Vet       | [Mm-vet: Evaluating large multimodal models for integrated capabilities.](https://arxiv.org/abs/2308.02490) |         [GitHub](https://github.com/yuweihao/MM-Vet)         |
|       ChEF        | [Chef: A comprehensive evaluation framework for standardized assessment of multimodal large language models.](https://arxiv.org/abs/2311.02692) |          [GitHub](https://openlamm.github.io/ChEF/)          |
|     Video-MME     | [Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis.](https://arxiv.org/abs/2405.21075) |    [GitHub](https://github.com/MME-Benchmarks/Video-MME)     |
|   MMBench-Video   | [Mmbench-video: A long-form multi-shot benchmark for holistic video understanding.](https://arxiv.org/abs/2406.14515) |          [GitHub](https://mmbench-video.github.io/)          |
|      MVBench      | [Mvbench: A comprehensive multi-modal video understanding benchmark.](https://arxiv.org/abs/2311.17005) | [Hugging Face](https://huggingface.co/datasets/OpenGVLab/MVBench) |
|  LongVideoBench   | [Longvideobench: A benchmark for long-context interleaved video-language understanding.](https://arxiv.org/abs/2407.15754) |  [GitHub](https://github.com/longvideobench/LongVideoBench)  |
|      LVBench      | [Lvbench: An extreme long video understanding benchmark.](https://arxiv.org/abs/2406.08035) |         [GitHub](https://github.com/zai-org/LVBench)         |
|    MotionBench    | [Motionbench: Benchmarking and improving fine-grained video motion understanding for vision language models.](https://arxiv.org/html/2501.02955v1) |       [GitHub](https://github.com/zai-org/MotionBench)       |
|    AudioBench     | [Audiobench: A universal benchmark for audio large language models.](https://arxiv.org/abs/2406.16020) |      [GitHub](https://github.com/AudioLLMs/AudioBench)       |
|     AIR-Bench     | [Air-bench: Benchmarking large audio-language models via generative comprehension.](https://arxiv.org/abs/2402.07729) |        [GitHub](https://github.com/OFA-Sys/AIR-Bench)        |
|  Dynamic-SUPERB   | [Dynamic-superb: Towards a dynamic, collaborative, and comprehensive instruction-tuning benchmark for speech.](https://arxiv.org/abs/2309.09510) |  [GitHub](https://github.com/dynamic-superb/dynamic-superb)  |
|     M3DBench      | [M3dbench: Let's instruct large models with multi-modal 3d prompts.](https://arxiv.org/abs/2312.10763) |        [GitHub](https://github.com/OpenM3D/M3DBench)         |
|        M3D        | [M3d: Advancing 3d medical image analysis with multi-modal large language models.](https://arxiv.org/abs/2404.00578) |          [GitHub](https://github.com/BAAI-DCAI/M3D)          |
|   Space3D-Bench   | [Space3d-bench: Spatial 3d question answering benchmark.](https://arxiv.org/abs/2408.16662) |       [Project Page](https://space3d-bench.github.io/)       |

#### General Knowledge

|  Benchmark  |                            Paper                             |                         Project Page                         |
| :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  ScienceQA  | [Learn to explain: Multimodal reasoning via thought chains for science question answering.](https://arxiv.org/abs/2209.09513) |       [GitHub](https://github.com/lupantech/ScienceQA)       |
|    CMMU     | [Cmmu: A benchmark for chinese multi-modal multi-type question understanding and reasoning.](https://arxiv.org/abs/2401.14011) |          [GitHub](https://github.com/FlagOpen/CMMU)          |
|  Scibench   | [Scibench: Evaluating college-level scientific problem-solving abilities of large language models.](https://arxiv.org/abs/2307.10635) |       [GitHub](https://github.com/mandyyyyii/scibench)       |
|   EXAMS-V   | [Exams-v: A multi-discipline multilingual multi-modal exam benchmark for evaluating vision language models.](https://arxiv.org/abs/2403.10378) | [Hugging Face](https://huggingface.co/datasets/Rocktim/EXAMS-V) |
|    MMMU     | [Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi.](https://arxiv.org/abs/2311.16502) |       [GitHub](https://github.com/MMMU-Benchmark/MMMU)       |
|  MMMU-Pro   | [Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark.](https://arxiv.org/abs/2409.02813) | [Hugging Face](https://huggingface.co/datasets/MMMU/MMMU_Pro/blob/main/README.md) |
|     HLE     | [Humanity's last exam.](https://arxiv.org/html/2501.14249v1) | [Project Page](https://otaheri.github.io/publication/2025_lastexam/) |
|    CURIE    | [Curie: Evaluating llms on multitask scientific long context understanding and reasoning.](https://arxiv.org/abs/2503.13517) |          [GitHub](https://github.com/google/curie)           |
|     SFE     | [Scientists' first exam: Probing cognitive abilities of mllm via perception, understanding, and reasoning.](https://arxiv.org/abs/2506.10521) |   [Hugging Face](https://huggingface.co/papers/2506.10521)   |
|    MMIE     | [Mmie: Massive multimodal interleaved comprehension benchmark for large vision-language models.](https://arxiv.org/abs/2410.10139) |        [GitHub](https://github.com/Lillianwei-h/MMIE)        |
| MDK12-Bench | [Mdk12-bench: A multi-discipline benchmark for evaluating reasoning in multimodal large language models.](https://arxiv.org/abs/2504.05782) |         [GitHub](https://github.com/LanceZPF/MDK12)          |
|    EESE     | [The ever-evolving science exam.](https://arxiv.org/abs/2507.16514) |       [Project Page](https://github.com/aiben-ch/EESE)       |

#### Safety

|  **Benchmark**  |                          **Paper**                           |                       **Project Page**                       |
| :-------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|     Unicorn     | [How many unicorns are in this image? a safety evaluation benchmark for vision llms.](https://arxiv.org/abs/2311.16101) | [GitHub](https://github.com/UCSC-VLAA/vllm-safety-benchmark) |
| JailbreakV-28K  | [Jailbreakv-28k: A benchmark for assessing the robustness of multimodal large language models against jailbreak attacks.](https://arxiv.org/abs/2404.03027) |   [GitHub](https://github.com/SaFoLab-WISC/JailBreakV_28K)   |
| MM-SafetyBench  | [Mm-safetybench: A benchmark for safety evaluation of multimodal large language models.](https://arxiv.org/abs/2311.17600) |     [GitHub](https://github.com/isXinLiu/MM-SafetyBench)     |
|    AVIBench     | [Avibench: Towards evaluating the robustness of large vision-language model on adversarial visual-instructions.](https://arxiv.org/abs/2403.09346) |     [GitHub](https://github.com/zhanghao5201/B-AVIBench)     |
|    MMJ-Bench    | [MMJ-Bench: A Comprehensive Study on Jailbreak Attacks and Defenses for Vision Language Models.](https://arxiv.org/abs/2408.08464) | [GitHub](https://github.com/thunxxx/MLLM-Jailbreak-evaluation-MMJ-bench) |
|       USB       | [Usb: A comprehensive and unified safety evaluation benchmark for multimodal large language models.](https://arxiv.org/abs/2505.23793) |    [GitHub](https://github.com/Hongqiong12/USB-SafeBench)    |
|    MLLMGuard    | [Mllmguard: A multi-dimensional safety evaluation suite for multimodal large language models.](https://arxiv.org/abs/2406.07594) |    [GitHub](https://github.com/Carol-gutianle/MLLMGuard)     |
|    SafeBench    | [Safebench: A safety evaluation framework for multimodal large language models.](https://arxiv.org/abs/2410.18927) |          [GitHub](https://safebench-mm.github.io/)           |
| MemeSafetyBench | [Are vision-language models safe in the wild? a meme-based benchmark study.](https://arxiv.org/abs/2505.15389) |   [Hugging Face](https://huggingface.co/papers/2505.15389)   |
|   UnsafeBench   | [Unsafebench: Benchmarking image safety classifiers on real-world and ai-generated images.](https://arxiv.org/abs/2405.03486) |                       *Not available*                        |
|      POPE       | [Evaluating object hallucination in large vision-language models.](https://arxiv.org/abs/2305.10355) |          [GitHub](https://github.com/RUCAIBox/POPE)          |
|   M-HalDetect   | [Detecting and preventing hallucinations in large vision language models.](https://arxiv.org/abs/2308.06394) |                       *Not available*                        |
|    Hal-Eval     | [Hal-eval: A universal and fine-grained hallucination evaluation framework for large vision language models.](https://arxiv.org/abs/2402.15721) | [GitHub](https://github.com/WisdomShell/hal-eval/tree/main)  |
|    Hallu-pi     | [Hallu-pi: Evaluating hallucination in multi-modal large language models within perturbed inputs.](https://arxiv.org/abs/2408.01355) |                       *Not available*                        |
|      BEAF       | [Beaf: Observing before-after changes to evaluate hallucination in vision-language models.](https://dl.acm.org/doi/10.1007/978-3-031-73247-8_14) |            [GitHub](https://beafbench.github.io/)            |
| HallusionBench  | [Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models.](https://arxiv.org/abs/2310.14566) | [Project Page](https://github.com/tianyi-lab/HallusionBench) |
|  AutoHallusion  | [Autohallusion: Automatic generation of hallucination benchmarks for vision-language models.](https://arxiv.org/abs/2406.10900) |   [GitHub](https://github.com/wuxiyang1996/AutoHallusion)    |
|   MultiTrust    | [Benchmarking trustworthiness of multimodal large language models: A comprehensive study.](https://arxiv.org/abs/2406.07057) |       [GitHub](https://github.com/thu-ml/MMTrustEval)        |
|      MMDT       | [Mmdt: Decoding the trustworthiness and safety of multimodal foundation models.](https://proceedings.iclr.cc/paper_files/paper/2025/file/0bcfb525c8f8f07ae10a93d0b2a40e00-Paper-Conference.pdf) |         [GitHub](https://github.com/AI-secure/MMDT)          |
|    Text2VLM     | [Text2vlm: Adapting text-only datasets to evaluate alignment training in visual language models.](https://arxiv.org/abs/2507.20704) |                       *Not available*                        |
|    MOSSBench    | [Mossbench: Is your multimodal language model oversensitive to safe queries?](https://arxiv.org/abs/2406.17806) |       [GitHub](https://github.com/xirui-li/MOSSBench)        |
|   CulturalVQA   | [Benchmarking vision language models for cultural understanding.](https://arxiv.org/abs/2407.10920) |           [Project Page](https://culturalvqa.org/)           |
|     ModScan     | [Modscan: Measuring stereotypical bias in large vision-language models from vision and language modalities.](https://arxiv.org/html/2410.06967v1) |  [GitHub](https://github.com/TrustAIRLab/ModSCAN/tree/main)  |
|     FMBench     | [Fmbench: Benchmarking fairness in multimodal large language models on medical tasks.](https://arxiv.org/abs/2410.01089) |                        Not available                         |
|    FairMedFM    | [Fairmedfm: Fairness benchmarking for medical imaging foundation models.](https://arxiv.org/abs/2407.00983) |       [GitHub](https://github.com/FairMedFM/FairMedFM)       |
|    FairCLIP     | [Fair-clip: Harnessing fairness in vision-language learning.](https://arxiv.org/abs/2403.19949) | [GitHub](https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP) |
|   DoxingBench   | [Doxing via the lens: Revealing privacy leakage in image geolocation for agentic multi-modal large reasoning model.](https://arxiv.org/abs/2504.19373) |   [Hugging Face](https://huggingface.co/papers/2504.19373)   |
|     PrivQA      | [Can language models be instructed to protect personal information?](https://arxiv.org/abs/2310.02224) |                       *Not available*                        |
|     SHIELD      | [Shield: An evaluation benchmark for face spoofing and forgery detection with multimodal large language models.](https://arxiv.org/abs/2402.04178) |       [GitHub](https://github.com/laiyingxin2/SHIELD)        |
|   ExtremeAIGC   | [Extremeaigc: Benchmarking lmm vulnerability to ai-generated extremist content.](https://arxiv.org/abs/2503.09964) |                       *Not available*                        |

### Specialized

#### Math

| **Benchmark**  |                          **Paper**                           |                       **Project Page**                       |
| :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   MathVista    | [Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts.](https://arxiv.org/abs/2310.02255) |       [GitHub](https://github.com/lupantech/MathVista)       |
|    PolyMATH    | [Polymath: A challenging multi-modal mathematical reasoning benchmark.](https://arxiv.org/abs/2410.14702) |     [Project Page](https://polymathbenchmark.github.io/)     |
|  MATH-Vision   | [Measuring multimodal mathematical reasoning with math-vision dataset.](https://arxiv.org/abs/2402.14804) |      [Project Page](https://mathvision-cuhk.github.io/)      |
| Olympiad-Bench | [Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems.](https://arxiv.org/abs/2402.14008) |      [GitHub](https://github.com/OpenBMB/OlympiadBench)      |
|    PolyMath    | [Polymath: Evaluating mathematical reasoning in multilingual contexts.](https://arxiv.org/abs/2504.18428) |         [GitHub](https://github.com/qwenlm/polymath)         |
|   Math-Verse   | [Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems?](https://arxiv.org/abs/2403.14624) |     [GitHub](https://github.com/ZrrSkywalker/MathVerse)      |
|    WE-MATH     | [We-math: Does your large multimodal model achieve human-like mathematical reasoning?](https://arxiv.org/abs/2407.01284) |                           `GitHub`                           |
|   MathScape    | [Mathscape: Evaluating mllms in multimodal math scenarios through a hierarchical benchmark.](https://arxiv.org/abs/2408.07543) |       [Github](https://github.com/Ahalfmoon/MathScape)       |
|    CMM-Math    | [Cmm-math: A chinese multimodal math dataset to evaluate and enhance the mathematics reasoning of large multimodal models.](https://arxiv.org/abs/2409.02834) | [Hugging Face](https://huggingface.co/datasets/ecnu-icalk/cmm-math) |
|    MV-MATH     | [Mv-math: Evaluating multimodal math reasoning in multi-visual contexts.](https://arxiv.org/abs/2502.20808) |       [GitHub](https://github.com/eternal8080/MV-MATH)       |

#### Physics

| **Benchmark** |                          **Paper**                           |                       **Project Page**                       |
| :-----------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   ScienceQA   | [Learn to explain: Multimodal reasoning via thought chains for science question answering.](https://arxiv.org/abs/2209.09513) |       [GitHub](https://github.com/lupantech/ScienceQA)       |
|      TQA      | [Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension.](https://openaccess.thecvf.com/content_cvpr_2017/papers/Kembhavi_Are_You_Smarter_CVPR_2017_paper.pdf) |                       *Not available*                        |
|     AI2D      | [A diagram is worth a dozen images.](https://arxiv.org/abs/1603.07396) |             [Project Page](https://allenai.org/)             |
|   MM-PhyQA    | [Mm-phyqa: Multimodal physics question answering with multi-image cot prompting.](https://arxiv.org/abs/2404.08704) |                       *Not available*                        |
| PhysUniBench  | [Physunibench: An undergraduate-level physics reasoning benchmark for multimodal models.](https://arxiv.org/abs/2506.17667) | [Project Page](https://prismax-team.github.io/PhysUniBenchmark/) |
| PhysicsArena  | [Physicsarena: The first multimodal physics reasoning benchmark exploring variable, process, and solution dimensions.](https://arxiv.org/abs/2505.15472) |   [Hugging Face](https://huggingface.co/papers/2505.15472)   |
|    SeePhys    | [Seephys: Does seeing help thinking? benchmarking vision-based physics reasoning.](https://arxiv.org/pdf/2505.19099) |         [GitHub](https://github.com/AI4Phys/SeePhys)         |
|  PhysReason   | [Physreason: A comprehensive benchmark towards physics-based reasoning.](https://arxiv.org/abs/2502.12054) | [Hugging Face](https://huggingface.co/datasets/zhibei1204/PhysReason) |
| OlympiadBench | [Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems.](https://arxiv.org/abs/2402.14008) |      [GitHub](https://github.com/OpenBMB/OlympiadBench)      |
|    SceMQA     | [Scemqa: A scientific college entrance level multimodal question answering benchmark.](https://arxiv.org/abs/2402.05138) |          [GitHub](https://github.com/SceMQA/SceMQA)          |
|     PACS      | [Pacs: A dataset for physical audiovisual commonsense reasoning.](https://arxiv.org/abs/2203.11130) |        [GitHub](https://github.com/samuelyu2002/PACS)        |
|     GRASP     | [GRASP: A novel benchmark for evaluating language grounding and situated physics understanding in multimodal language models.](https://arxiv.org/abs/2311.09048) |      [GitHub](https://github.com/i-machine-think/grasp)      |
|   CausalVQA   | [Causalvqa: A physically grounded causal reasoning benchmark for video models.](https://arxiv.org/abs/2506.09943) |   [GitHub](https://github.com/facebookresearch/CausalVQA)    |
|    LiveXiv    | [Livexiv a multi-modal live benchmark based on arxiv papers content.](https://arxiv.org/abs/2410.10783) |      [GitHub](https://github.com/NimrodShabtay/LiveXiv)      |

#### Chemistry

| **Benchmark** |                          **Paper**                           |                       **Project Page**                       |
| :-----------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    SMILES     | [Smiles, a chemical language and information system. 1. introduction to methodology and encoding rules.](https://pubs.acs.org/doi/abs/10.1021/ci00057a005) |            [Project Page](http://opensmiles.org/)            |
|   ChEBI-20    | [Text2mol: Cross-modal molecule retrieval with natural language queries.](https://github.com/cnedwards/text2mol) |       [GitHub](https://github.com/cnedwards/text2mol)        |
|   ChemBench   | [Chemllm: A chemical large language model.](https://arxiv.org/abs/2402.06852) | [GitHub](https://github.com/ChemFoundationModels/ChemLLMBench) |
|    SELFIES    | [Self-referencing embedded strings (selfies): A 100% robust molecular string representation.](https://github.com/aspuru-guzik-group/selfies) |   [GitHub](https://github.com/aspuru-guzik-group/selfies)    |
|     InChI     | [Inchi, the iupac international chemical identifier.](https://iupac.org/who-we-are/divisions/division-details/inchi/) |         [Project Page](https://www.inchi-trust.org/)         |
|     MolX      | [Molx: Enhancing large language models for molecular learning with a multi-modal extension.](https://arxiv.org/abs/2406.06777) |                       *Not available*                        |
|    GiT-Mol    | [Git-mol: A multi-modal large language model for molecular science with graph, image, and text.](https://arxiv.org/abs/2308.06911) |  [GitHub](https://github.com/AI-HPC-Research-Team/GIT-Mol)   |
| Instruct-Mol  | [Instructmol: Multi-modal integration for building a versatile and reliable molecular assistant in drug discovery.](https://arxiv.org/abs/2311.16208) |       [GitHub](https://idea-xl.github.io/InstructMol/)       |
|  ChEBI-20-MM  | [A quantitative analysis of knowledge-learning preferences in large language models in molecular science.](https://arxiv.org/abs/2402.04119v2) |  [GitHub](https://github.com/AI-HPC-Research-Team/SLM4Mol)   |
|  MMCR-Bench   | [Chemvlm: Exploring the power of multimodal large language models in chemistry area.](https://arxiv.org/abs/2408.07246) |  [GitHub](https://github.com/jiaqingxie/ChemVLM-Mechanism)   |
|   MACBench    | [Probing the limitations of multimodal language models for chemistry and materials research.](https://www.researchgate.net/publication/386144119_Probing_the_limitations_of_multimodal_language_models_for_chemistry_and_materials_research) |      [GitHub](https://github.com/lamalab-org/macbench)       |
|    3D-MoLM    | [Towards 3d molecule-text interpretation in language models.](https://arxiv.org/abs/2401.13923) |         [GitHub](https://github.com/lsh0520/3D-MoLM)         |
|    M3-20M     | [M3-20m: A large-scale multi-modal molecule dataset for ai-driven drug design and discovery.](https://arxiv.org/abs/2412.06847v2) |           [GitHub](https://github.com/bz99bz/M-3)            |
|  MassSpecGym  | [Massspecgym: A benchmark for the discovery and identification of molecules.](https://arxiv.org/abs/2410.23326v1) | [GitHub](https://polarishub.io/datasets/roman-bushuiev/massspecgym) |
|   MolPuzzle   | [Can llms solve molecule puzzles? a multimodal benchmark for molecular structure elucidation.](https://kehanguo2.github.io/Molpuzzle.io/paper/SpectrumLLM__Arxiv_.pdf) |     [GitHub](https://kehanguo2.github.io/Molpuzzle.io/)      |

#### Finance

|   Benchmark    |                            Paper                             |                         Project Page                         |
| :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|     FinMME     | [FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation](https://arxiv.org/abs/2505.24714) |        [Github](https://github.com/luo-junyu/FinMME)         |
|     FAMMA      | [FAMMA: A Benchmark for Financial Domain Multilingual Multimodal Question Answering](https://arxiv.org/abs/2410.04526) |     [Project Page](https://famma-bench.github.io/famma/)     |
|  MME-Finance   | [MME-Finance: A Multimodal Finance Benchmark for Expert-level Understanding and Reasoning](https://arxiv.org/abs/2411.03314) | [Project Page](https://hithink-research.github.io/MME-Finance/) |
|  MultiFinBen   | [MultiFinBen: A Comprehensive Multimodal Financial Benchmark](https://arxiv.org/abs/2506.14028) |     [Github](https://github.com/xueqingpeng/MultiFinBen)     |
| CFBenchmark-MM | [CFBenchmark-MM: A Comprehensive Multimodal Financial Benchmark](https://www.arxiv.org/abs/2506.13055) |    [Github](https://github.com/TongjiFinLab/CFBenchmark)     |
|     FinMMR     | [FinMMR: Multimodal Financial Reasoning Benchmark](https://arxiv.org/abs/2508.04625v1) |    [Github](https://github.com/BUPT-Reasoning-Lab/FinMMR)    |
|    Fin-Fact    | [Fin-Fact: Financial Fact Checking Dataset](https://arxiv.org/abs/2309.08793) |        [Github](https://github.com/IIT-DM/Fin-Fact/)         |
|      FCMR      | [FCMR: Financial Multimodal Reasoning](https://arxiv.org/abs/2412.12567) |                                                              |
|    FinTral     | [FinTral: Financial Translation and Analysis](https://arxiv.org/abs/2402.10986) |         [Github](https://github.com/UBC-NLP/fintral)         |
|  Open-FinLLMs  | [Open-FinLLMs: Open Financial Large Language Models](https://arxiv.org/abs/2408.11878) | [Hugging Face](https://huggingface.co/collections/TheFinAI/open-finllms-66b671f2b4958a65e20decbe) |
|    FinGAIA     | [FinGAIA: Financial AI Assistant](https://arxiv.org/abs/2507.17186) |     [Github](https://github.com/SUFE-AIFLM-Lab/FinGAIA)      |

#### Healthcare & Medical Science

|      Benchmark      |                            Paper                             |                         Project Page                         |
| :-----------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|       VQA-RAD       | [VQA-RAD: Visual Question Answering Radiology Dataset](https://www.nature.com/articles/sdata2018251) | [Project Page](https://www.nature.com/articles/sdata2018251) |
|       PathVQA       | [PathVQA: Pathology Visual Question Answering](https://arxiv.org/abs/2003.10286) |      [Github](https://github.com/KaveeshaSilva/PathVQA)      |
|     RP3D-DiagDS     | [RP3D-DiagDS: 3D Medical Diagnosis Dataset](https://arxiv.org/abs/2312.16151) |  [Project Page](https://qiaoyu-zheng.github.io/RP3D-Diag/)   |
|      PubMedQA       | [PubMedQA: Medical Question Answering Dataset](https://arxiv.org/abs/1909.06146) |         [Project Page](https://pubmedqa.github.io/)          |
|     HealthBench     | [HealthBench: Medical AI Benchmark](https://arxiv.org/abs/2505.08775) |         [Project Page](https://www.healthbench.co/)          |
|    GMAI-MMBench     | [GMAI-MMBench: General Medical AI Multimodal Benchmark](https://arxiv.org/abs/2408.03361) | [Project Page](https://uni-medical.github.io/GMAI-MMBench.github.io/#2023xtuner) |
|   OpenMM-Medical    | [OpenMM-Medical: Open Medical Multimodal Model](https://arxiv.org/abs/2501.15368) | [Github](https://github.com/baichuan-inc/Baichuan-Omni-1.5)  |
| Genomics-Long-Range | [Genomics-Long-Range: Long-Range Genomic Benchmark](https://openreview.net/forum?id=8O9HLDrmtq) | [Hugging Face](https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark) |
|    Genome-Bench     | [Genome-Bench: Comprehensive Genomics Benchmark](https://arxiv.org/abs/2505.19501v1) | [Hugging Face](https://huggingface.co/datasets/Mingyin0312/Genome-Bench) |

#### Code

|       Benchmark       |                            Paper                             |                         Project Page                         |
| :-------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|      Design2Code      | [Design2Code: From Design Mockups to Code](https://arxiv.org/abs/2403.03163) |   [Project Page](https://salt-nlp.github.io/Design2Code/)    |
|       Web2Code        | [Web2Code: Web-to-Code Generation](https://arxiv.org/abs/2406.20098) |  [Project Page](https://mbzuai-llm.github.io/webpage2code/)  |
|       Plot2Code       | [Plot2Code: From Charts to Code](https://arxiv.org/abs/2405.07990) | [Hugging Face](https://huggingface.co/datasets/TencentARC/Plot2Code) |
|      ChartMimic       | [ChartMimic: Chart Understanding and Generation](https://arxiv.org/abs/2406.09961) |        [Project Page](https://chartmimic.github.io/)         |
|      HumanEval-V      | [HumanEval-V: Visual Code Generation Benchmark](https://arxiv.org/abs/2410.12381) |        [Project Page](https://humaneval-v.github.io/)        |
|      Code-Vision      | [Code-Vision: Visual Code Understanding](https://arxiv.org/abs/2502.11829) |   [Github](https://github.com/wanghanbinpanda/CodeVision)    |
| SWE-bench Multi-modal | [SWE-bench Multi-modal: Software Engineering Benchmark](https://arxiv.org/abs/2410.03859) |   [Project Page](https://www.swebench.com/multimodal.html)   |
|        MMCode         | [MMCode: Multimodal Code Generation](https://arxiv.org/abs/2404.09486) |       [Github](https://github.com/likaixin2000/MMCode)       |
|        M²Eval         | [M²Eval: Multimodal Code Evaluation](https://arxiv.org/abs/2507.08719) |         [Github](https://github.com/MCEVAL/MMCoder)          |
|     BigDocs-Bench     | [BigDocs-Bench: Large Document Understanding](https://arxiv.org/abs/2412.04626) |          [Project Page](https://bigdocs.github.io/)          |
|     BigDocs-Bench     | [Bigdocs: An open dataset for training multimodal models on document and code tasks.](https://arxiv.org/abs/2412.04626) |        [GitHub](https://github.com/Big-Docs/BigDocs)         |

### Autonomous Driving

|  Benchmark   |                            Paper                             |                         Project Page                         |
| :----------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  Rank2Tell   | [Rank2Tell: Ranking-based Visual Storytelling](https://arxiv.org/abs/2309.06597) |      [Project Page](https://usa.honda-ri.com/rank2tell)      |
|    DRAMA     | [DRAMA: Dynamic Risk Assessment for Autonomous Vehicles](https://arxiv.org/abs/2209.10767) |        [Project Page](https://usa.honda-ri.com/drama)        |
| NuScenes-QA  | [NuScenes-QA: Autonomous Driving Question Answering](https://arxiv.org/abs/2305.14836) |     [Github](https://github.com/qiantianwen/NuScenes-QA)     |
|   LingoQA    | [LingoQA: Driving Language Understanding](https://arxiv.org/abs/2312.14115) |         [Github](https://github.com/wayveai/LingoQA)         |
|   V2V-LLM    | [V2V-LLM: Vehicle-to-Vehicle Communication](https://arxiv.org/abs/2502.09980) | [Project Page](https://eddyhkchiu.github.io/v2vllm.github.io/) |
|   MAPLM-QA   | [MAPLM: Real-World Large-Scale Vision-Language Benchmark for Map and Traffic](https://openaccess.thecvf.com/content/CVPR2024/papers/Cao_MAPLM_A_Real-World_Large-Scale_Vision-Language_Benchmark_for_Map_and_Traffic_CVPR_2024_paper.pdf) |          [Github](https://github.com/LLVM-AD/MAPLM)          |
|    SURDS     | [SURDS: Autonomous Driving Dataset](https://github.com/XiandaGuo/Drive-MLLM) |      [Github](https://github.com/XiandaGuo/Drive-MLLM)       |
|  AD2-Bench   | [AD2-Bench: Autonomous Driving Benchmark](https://arxiv.org/abs/2506.09557) |                                                              |
| DriveAction  | [DriveAction: Driving Action Recognition](https://arxiv.org/abs/2506.05667v1) | [Hugging Face](https://huggingface.co/datasets/LiAuto-DriveAction/drive-action) |
| DriveLMM-01  | [DriveLMM-01: Driving Language Model](https://arxiv.org/abs/2503.10621) |     [Github](https://github.com/mbzuai-oryx/DriveLMM-o1)     |
|   DriveVLM   | [DriveVLM: Vision-Language Model for Driving](https://arxiv.org/abs/2402.12289) | [Project Page](https://tsinghua-mars-lab.github.io/DriveVLM/) |
| RoboTron-Sim | [RoboTron-Sim: Robot Simulation Platform](https://www.arxiv.org/abs/2508.04642) |  [Project Page](https://stars79689.github.io/RoboTron-Sim/)  |
|     IDKB     | [IDKB: Intelligent Driving Knowledge Base](https://arxiv.org/abs/2409.02914) | [Project Page](https://4dvlab.github.io/project_page/idkb.html) |
|  VLADBench   | [VLADBench: Vision-Language-Action Driving Benchmark](https://arxiv.org/abs/2503.21505) |      [Github](https://github.com/Depth2World/VLADBench)      |
|    DriVQA    | [DriVQA: Driving Visual Question Answering](https://www.sciencedirect.com/science/article/pii/S235234092500099X) |                                                              |

#### Earth Science / Remote Sensing

|     Benchmark     |                            Paper                             |                         Project Page                         |
| :---------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   GEOBench-VLM    | [GEOBench-VLM: Geospatial Vision-Language Model Benchmark](https://arxiv.org/abs/2411.19325) | [Project Page](http://the-ai-alliance.github.io/GEO-Bench-VLM) |
|      ClimaQA      | [ClimaQA: Climate Question Answering](https://arxiv.org/abs/2410.16701) |   [Github](https://github.com/Rose-STL-Lab/genie-climaqa)    |
|    ClimateBERT    | [ClimateBERT: Climate Language Model](https://arxiv.org/abs/2110.12010) |    [Project Page](https://www.chatclimate.ai/climatebert)    |
|     WeatherQA     | [WeatherQA: Weather Question Answering](https://arxiv.org/abs/2406.11217) |      [Github](https://github.com/chengqianma/WeatherQA)      |
|    OceanBench     | [OceanBench: Ocean Data Analysis Benchmark](https://arxiv.org/abs/2309.15599) | [Project Page](https://jejjohnson.github.io/oceanbench/content/overview.html) |
|  OmniEarth-Bench  | [OmniEarth-Bench: Comprehensive Earth Observation](https://arxiv.org/abs/2505.23522) | [Hugging Face](https://huggingface.co/datasets/initiacms/OmniEarth-Bench) |
|      MSEarth      | [MSEarth: Multi-Scale Earth Observation](https://arxiv.org/abs/2505.20740) |       [Github](https://github.com/xiangyu-mm/MSEarth)        |
|      EarthSE      | [EarthSE: Earth System Evaluation](https://www.arxiv.org/abs/2505.17139) |       [Hugging Face](https://huggingface.co/ai-earth)        |
|       RSICD       | [RSICD: Remote Sensing Image Captioning Dataset](https://arxiv.org/abs/1712.07835) |  [Github](https://github.com/201528014227051/RSICD_optimal)  |
|   NWPU-Captions   | [NWPU-Captions: Remote Sensing Image Descriptions](https://arxiv.org/abs/2402.06475v1) |   [Github](https://github.com/HaiyanHuang98/NWPU-Captions)   |
| RSVQA-HRBEN/LRBEN | [RSVQA: Remote Sensing Visual Question Answering](https://arxiv.org/abs/2003.07333) |       [Project Page](https://rsvqa.sylvainlobry.com/)        |
|     DIOR-RSVG     | [DIOR-RSVG: Remote Sensing Visual Grounding](https://arxiv.org/pdf/2210.12634) |   [Github](https://github.com/ZhanYang-nwpu/RSVG-pytorch)    |
|     VRSBench      | [VRSBench: Visual Remote Sensing Benchmark](https://arxiv.org/abs/2406.12384) |         [Project Page](https://vrsbench.github.io/)          |
|      LRS-VQA      | [LRS-VQA: Large-scale Remote Sensing VQA](https://arxiv.org/abs/2503.07588) |       [Github](https://github.com/VisionXLab/LRS-VQA)        |
|   GeoChat-Bench   | [GeoChat-Bench: Geospatial Conversation Benchmark](https://arxiv.org/pdf/2311.15826) |       [Github](https://github.com/mbzuai-oryx/geochat)       |
|    XLRS-Bench     | [XLRS-Bench: Cross-Lingual Remote Sensing](https://arxiv.org/abs/2503.23771) |        [Project Page](https://xlrs-bench.github.io/)         |
|      RSIEval      | [RSIEval: Remote Sensing Image Evaluation](https://arxiv.org/abs/2307.15266) |        [Github](https://github.com/Lavender105/RSGPT)        |
|      UrBench      | [UrBench: Urban Remote Sensing Benchmark](https://arxiv.org/abs/2408.17267) |    [Project Page](https://opendatalab.github.io/UrBench/)    |
|      CHOICE       | [CHOICE: Comprehensive Remote Sensing Benchmark](https://arxiv.org/abs/2411.18145) |       [Github](https://github.com/ShawnAn-WHU/CHOICE)        |
| SARChat-Bench-2M  | [SARChat-Bench-2M: SAR Image Understanding](https://arxiv.org/abs/2502.08168) |        [Github](https://github.com/JimmyMa99/SARChat)        |
|    LHRS-Bench     | [LHRS-Bench: Large-scale High-Resolution Remote Sensing](https://arxiv.org/abs/2402.02544) |        [Github](https://github.com/NJU-LHRS/LHRS-Bot)        |
|     FIT-RSFG      | [FIT-RSFG: Remote Sensing Fine-Grained Recognition](https://arxiv.org/abs/2406.10100) |       [Github](https://github.com/Luo-Z13/SkySenseGPT)       |
|    VLEO-Bench     | [VLEO-Bench: Very Low Earth Orbit Benchmark](https://arxiv.org/abs/2401.17600) |           [Project Page](https://vleo.danielz.ch/)           |
|     NAIP-OSM      | [NAIP-OSM: Aerial Imagery and Map Alignment](https://arxiv.org/pdf/2110.04690) |          [Project Page](https://favyen.com/muno21/)          |

#### Embodied Tasks

|                    Benchmark                    |                            Paper                             |                         Project Page                         |
| :---------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|      Embodied Questioning Answering (EQA)       | [EXPRESS-Bench: Embodied Question Answering](https://arxiv.org/abs/2503.11117) |           [Project Page](https://embodiedqa.org/)            |
|               R2R (Room-to-Room)                | [R2R: Room-to-Room Navigation](https://arxiv.org/abs/1711.07280) |          [Github](https://github.com/xjli/r2r_vln)           |
|                     Reverie                     | [Reverie: Remote Embodied Visual Referring Expression](https://arxiv.org/abs/1904.10151) | [Project Page](https://yuankaiqi.github.io/REVERIE_Challenge/dataset.html) |
|                     Alfred                      | [Alfred: A Benchmark for Interpreting Grounded Instructions](https://arxiv.org/abs/1912.01734) |       [Github](https://github.com/askforalfred/alfred)       |
|                     Calvin                      | [Calvin: Long-Horizon Language-Conditioned Robot Learning](https://arxiv.org/abs/2112.03227) |           [Github](https://github.com/mees/calvin)           |
|                  EPIC-KITCHENS                  | [EPIC-KITCHENS: Large Scale Dataset in First Person Vision](https://arxiv.org/abs/2005.00343) |       [Project Page](https://epic-kitchens.github.io/)       |
|                      Ego4D                      | [Ego4D: Around the World in 3,000 Hours](https://arxiv.org/abs/2110.07058) |           [Project Page](https://ego4d-data.org/)            |
|                      EMQA                       | [EMQA: Ego-centric Multimodal Question Answering](https://arxiv.org/abs/2205.01652) |        [Github](https://github.com/lbaermann/qaego4d)        |
|                      SQA3D                      | [SQA3D: Situated Question Answering in 3D Scenes](https://arxiv.org/abs/2210.07474) |           [Project Page](https://sqa3d.github.io/)           |
|                    Open-EQA                     | [Open-EQA: Open-Vocabulary Embodied Question Answering](https://open-eqa.github.io/assets/pdfs/paper.pdf) |         [Project Page](https://open-eqa.github.io/)          |
|                     HM-EQA                      | [HM-EQA: Hierarchical Multi-modal Embodied QA](https://arxiv.org/abs/2403.15941v1) |        [Project Page](https://explore-eqa.github.io/)        |
|                      MOTIF                      | [MOTIF: Multimodal Object-Text Interaction Framework](https://arxiv.org/abs/2502.12479) |       [Github](https://github.com/blt2114/MotifBench)        |
|                    EgoTaskQA                    | [EgoTaskQA: Understanding Tasks in Egocentric Videos](https://arxiv.org/abs/2210.03929) |   [Project Page](https://sites.google.com/view/egotaskqa)    |
|                  EmbodiedScan                   | [EmbodiedScan: Holistic Multi-Modal 3D Perception](https://arxiv.org/abs/2312.16170) |   [Project Page](https://tai-wang.github.io/embodiedscan/)   |
|                     RH20T-P                     | [RH20T-P: Robotic Manipulation Dataset](https://arxiv.org/abs/2403.19622) |           [Project Page](https://rh20t.github.io/)           |
|                  EXPRESS-Bench                  | [EXPRESS-Bench: Embodied Question Answering](https://arxiv.org/abs/2503.11117) |    [Github](https://github.com/HCPLab-SYSU/EXPRESS-Bench)    |
|                  EmbodiedEval                   | [EmbodiedEval: Embodied AI Evaluation](https://arxiv.org/abs/2501.11858) |       [Project Page](https://embodiedeval.github.io/)        |
|                 Embodied Bench                  | [Embodied Bench: Comprehensive Embodied AI Evaluation](https://arxiv.org/abs/2502.09560v1) |       [Project Page](https://embodiedbench.github.io/)       |
|                    VLABench                     | [VLABench: Vision-Language-Action Benchmark](https://arxiv.org/abs/2412.18194) |         [Project Page](https://vlabench.github.io/)          |
|                    EWMBench                     | [EWMBench: Embodied World Model Benchmark](https://arxiv.org/abs/2505.09694) |       [Github](https://github.com/AgibotTech/EWMBench)       |
| NeurIPS 2025 Embodied Agent Interface Challenge |       NeurIPS 2025 Embodied Agent Interface Challenge        |       [Project Page](https://neurips25-eai.github.io/)       |

## Generation Evaluation

### Image

|                      Benchmark                       |                            Paper                             |                         Project Page                         |
| :--------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                     DiffusionDB                      | [DiffusionDB: A Large-scale Prompt Gallery Dataset for Text-to-Image Generative Models](https://arxiv.org/pdf/2210.14896) |      [GitHub](https://poloclub.github.io/diffusiondb/)       |
|                   HPD、HPD v2、HPS                   | [Human Preference Score: Better Aligning Text-to-Image Models with Human Preference](https://arxiv.org/pdf/2303.14420) |      [GitHub](https://tgxs002.github.io/align_sd_web/)       |
|            ImageReward、ImageReward/ReFL             | [ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/pdf/2304.05977) |       [GitHub](https://github.com/zai-org/ImageReward)       |
|                Pick-A-Pic、PickScore                 | [Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation](https://arxiv.org/pdf/2305.01569) |     [GitHub](https://github.com/yuvalkirstain/PickScore)     |
|                       AGIQA-1K                       | [A Perceptual Quality Assessment Exploration for AIGC Images](https://arxiv.org/pdf/2303.12618) | [GitHub](https://github.com/lcysyzxdxc/AGIQA-1k-Database/tree/main) |
|                       AGIQA-3K                       | [AGIQA-3K: An Open Database for AI-Generated Image Quality Assessment](https://arxiv.org/pdf/2306.04717) |  [GitHub](https://github.com/lcysyzxdxc/AGIQA-3k-Database)   |
|                     AIGCIQA2023                      | [AIGCIQA2023: A Large-scale Image Quality Assessment Database for AI Generated Images: from the Perspectives of Quality, Authenticity and Correspondence](https://arxiv.org/pdf/2307.00211) | [Hugging Face](https://huggingface.co/datasets/strawhat/aigciqa2023) |
|                     AGIN、JOINT                      | [Exploring the Naturalness of AI-Generated Images](https://arxiv.org/pdf/2312.05476) | [GitHub](https://github.com/zijianchen98/AGIN?utm_source=catalyzex.com) |
|                      AIGIQA-20K                      | [AIGIQA-20K: A Large Database for AI-Generated Image Quality Assessment](https://arxiv.org/pdf/2404.03407) | [Hugging Face](https://huggingface.co/datasets/strawhat/aigciqa-20k) |
|                     AIGCOIQA2024                     | [AIGCOIQA2024: Perceptual Quality Assessment of AI Generated Omnidirectional Images](https://arxiv.org/pdf/2404.01024) |       [GitHub](https://github.com/IntMeGroup/AIGCOIQA)       |
|                      CMC-Bench                       | [CMC-Bench: Towards a New Paradigm of Visual Signal Compression](https://arxiv.org/pdf/2406.09356) |       [GitHub](https://github.com/Q-Future/CMC-Bench)        |
|               PKU-I2IQA、NR/FR-AIGCIQA               | [PKU-I2IQA: An Image-to-Image Quality Assessment Database for AI Generated Images](https://arxiv.org/pdf/2311.15556) |         [GitHub](https://github.com/jiquan123/I2IQA)         |
|                       SeeTRUE                        | [What You See is What You Read? Improving Text-Image Alignment Evaluation](https://arxiv.org/pdf/2305.10400) |           [GitHub](https://wysiwyr-itm.github.io/)           |
|                AIGCIQA2023+、MINT-IQA                | [Quality Assessment for AI Generated Images with Instruction Tuning](https://arxiv.org/pdf/2405.07346) |       [GitHub](https://github.com/IntMeGroup/MINT-IQA)       |
|              Q-Eval-100K、Q-Eval Score               | [Q-Eval-100K: Evaluating Visual Quality and Alignment Level for Text-to-Vision Content](https://arxiv.org/pdf/2503.02357) |         [GitHub](https://github.com/zzc-1998/Q-Eval)         |
| Measuring the Quality of Text-to-Video Model Outputs | [Measuring the Quality of Text-to-Video Model Outputs: Metrics and Dataset](https://arxiv.org/pdf/2309.08009) | [Dataset Download](https://figshare.com/articles/dataset/Text_prompts_and_videos_generated_using_5_popular_Text-to-Video_models_plus_quality_metrics_including_user_quality_assessments/24078045) |
|                     EvalCrafter                      | [EvalCrafter: Benchmarking and Evaluating Large Video Generation Models](https://arxiv.org/pdf/2310.11440) |           [GitHub](https://evalcrafter.github.io/)           |
|                         FETV                         | [FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain Text-to-Video Generation](https://arxiv.org/pdf/2311.01813) |           [GitHub](https://github.com/llyx97/FETV)           |
|                        VBench                        | [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://arxiv.org/pdf/2311.17982) |     [GitHub](https://vchitect.github.io/VBench-project/)     |
|                       T2VQA-DB                       | [Subjective-Aligned Dataset and Metric for Text-to-Video Quality](https://arxiv.org/pdf/2403.11956) |           [GitHub](https://github.com/QMME/T2VQA)            |
|                         GAIA                         | [GAIA: Rethinking Action Quality Assessment for AI-Generated Videos](https://arxiv.org/pdf/2406.06087) | [GitHub](https://github.com/zijianchen98/GAIA?utm_source=catalyzex.com) |
|               AIGVQA-DB、AIGV-Assessor               | [AIGV-Assessor: Benchmarking and Evaluating the Perceptual Quality of Text-to-Video Generation with LMM](https://arxiv.org/pdf/2411.17221) |   [GitHub](https://github.com/wangjiarui153/AIGV-Assessor)   |
|                   AIGVE-60K、LOVE                    | [LOVE: Benchmarking and Evaluating Text-to-Video Generation and Video-to-Text Interpretation](https://arxiv.org/pdf/2505.12098) |         [GitHub](https://github.com/IntMeGroup/LOVE)         |
|                 Human-AGVQA-DB、GHVQ                 | [Human-Activity AGV Quality Assessment: A Benchmark Dataset and an Objective Evaluation Metric](https://arxiv.org/pdf/2411.16619) |        [GitHub](https://github.com/zczhang-sjtu/GHVQ)        |
|                TDVE-DB、TDVE-Assessor                | [TDVE-Assessor: Benchmarking and Evaluating the Quality of Text-Driven Video Editing with LMMs](https://arxiv.org/pdf/2505.19535) |    [GitHub](https://github.com/JuntongWang/TDVE-Assessor)    |
|                AGAVQA-3K、AGAV-Rater                 | [AGAV-Rater: Adapting Large Multimodal Model for AI-Generated Audio-Visual Quality Assessment](https://arxiv.org/pdf/2501.18314) |    [GitHub](https://github.com/charlotte9524/AGAV-Rater)     |
|                      Qwen-ALLD                       | [Audio Large Language Models Can Be Descriptive Speech Quality Evaluators](https://arxiv.org/pdf/2501.17202) | [Hugging Face](https://huggingface.co/datasets/PeacefulData/speech-quality-descriptive-caption) |
|                       BASE-TTS                       | [BASE TTS: Lessons from building a billion-parameter Text-to-Speech model on 100K hours of data](https://arxiv.org/pdf/2402.08093) | [Audio samples  of BASE-TTS](https://www.amazon.science/base-tts-samples) |
|                         ATT                          | [Audio Turing Test: Benchmarking the Human-likeness of Large Language Model-based Text-to-Speech Systems in Chinese](https://arxiv.org/pdf/2505.11200) | [Hugging Face](https://huggingface.co/collections/meituan/audio-turing-test-682446320368164faeaf38a4) |
|                        TTSDS2                        | [TTSDS2: Resources and Benchmark for Evaluating Human-Quality Text to Speech Systems](https://arxiv.org/pdf/2506.19441) |            [Website](https://ttsdsbenchmark.com/)            |
|                    MATE-3D、LGVQ                     | [Benchmarking and Learning Multi-Dimensional Quality Evaluator for Text-to-3D Generation](https://arxiv.org/pdf/2412.11170) |             [GitHub](https://mate-3d.github.io/)             |
|                        3DGCQA                        | [3DGCQA: A Quality Assessment Database for 3D AI-Generated Contents](https://arxiv.org/pdf/2409.07236) |         [GitHub](https://github.com/zyj-2000/3DGCQA)         |
|                     AIGC-T23DAQA                     | [Multi-Dimensional Quality Assessment for Text-to-3D Assets: Dataset and Model](https://arxiv.org/pdf/2502.16915) | [GitHub](https://github.com/ZedFu/T23DAQA?utm_source=catalyzex.com) |
|                       SI23DCQA                       | [SI23DCQA: Perceptual Quality Assessment of Single Image-to-3D Content](https://jhc.sjtu.edu.cn/~xiaohongliu/papers/2025SI23DCQA.pdf) |         [GitHub](https://github.com/ZedFu/SI23DCQA)          |
|                    3DGS-IEval-15K                    | [3DGS-IEval-15K: A Large-scale Image Quality Evaluation Database for 3D Gaussian-Splatting](https://arxiv.org/pdf/2506.14642) |    [GitHub ](https://github.com/YukeXing/3DGS-IEval-15K)     |
|                 Inception Score (IS)                 | [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498) |       [GitHub](https://github.com/openai/improved-gan)       |
|                         FVD                          | [Towards Accurate Generative Models of Video: A New Metric & Challenges](https://arxiv.org/pdf/1812.01717) | [GitHub](https://github.com/google-research/google-research/tree/master/frechet_video_distance) |
|                       VQAScore                       | [Evaluating Text-to-Visual Generation with Image-to-Text Generation](https://arxiv.org/pdf/2404.01291) |    [GitHub](https://linzhiqiu.github.io/papers/vqascore/)    |
|                  NTIRE 2024 AIGC QA                  | [NTIRE 2024 Quality Assessment of AI-Generated Content Challenge](https://arxiv.org/pdf/2404.16687) |           [Website](https://cvlai.net/ntire/2024/)           |
|                       Q-Bench                        | [Q-Bench: A Benchmark for General-Purpose Foundation Models on Low-level Vision](https://arxiv.org/pdf/2309.14181) |        [GitHub](https://q-future.github.io/Q-Bench/)         |
|                      Q-instruct                      | [Q-Instruct: Improving Low-level Visual Abilities for Multi-modality Foundation Models](https://arxiv.org/pdf/2311.06783) |       [GitHub](https://q-future.github.io/Q-Instruct/)       |
|                       Q-align                        | [Q-ALIGN: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels](https://arxiv.org/pdf/2312.17090) |        [GitHub](https://github.com/Q-Future/Q-Align)         |
|                       Q-boost                        | [Q-Boost: On Visual Quality Assessment Ability of Low-level Multi-Modality Foundation Models](https://arxiv.org/pdf/2312.15300) | [Project Page](https://q-future.github.io/Q-Instruct/boost_qa/) |
|                     Co-Instruct                      | [Towards Open-ended Visual Quality Comparison](https://arxiv.org/pdf/2402.16641) | [Hugging Face](https://huggingface.co/q-future/co-instruct)  |
|                       DepictQA                       | [Depicting Beyond Scores: Advancing Image Quality Assessment through Multi-modal Language Models](https://arxiv.org/pdf/2312.08962) |            [GitHub](https://depictqa.github.io/)             |
|                       M3-AGIQA                       | [M3-AGIQA: Multimodal, Multi-Round, Multi-Aspect AI-Generated Image Quality Assessment](https://arxiv.org/pdf/2502.15167) |      [GitHub](https://github.com/strawhatboy/M3-AGIQA)       |
|                       Q-Refine                       | [Q-Refine: A Perceptual Quality Refiner for AI-Generated Image](https://arxiv.org/pdf/2401.01117) | [GitHub](https://github.com/Q-Future/Q-Refine?utm_source=catalyzex.com) |
|                        AGIQA                         | [Large Multi-modality Model Assisted AI-Generated Image Quality Assessment](https://arxiv.org/pdf/2404.17762) |        [GitHub](https://github.com/wangpuyi/MA-AGIQA)        |
|                        SF-IQA                        | [SF-IQA: Quality and Similarity Integration for AI Generated Image Quality Assessment](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Yu_SF-IQA_Quality_and_Similarity_Integration_for_AI_Generated_Image_Quality_CVPRW_2024_paper.pdf) |          [GitHub](https://github.com/Yvzhh/SF-IQA)           |
|                       SC-AGIQA                       | [Text-Visual Semantic Constrained AI-Generated Image Quality Assessment](https://arxiv.org/pdf/2507.10432) |         [GitHub](https://github.com/mozhu1/SC-AGIQA)         |
|                       TSP-MGS                        | [AI-Generated Image Quality Assessment Based on Task-Specific Prompt and Multi-Granularity Similarity](https://arxiv.org/pdf/2411.16087) |                       *Not available*                        |
|                      MoE-AGIQA                       | [MoE-AGIQA: Mixture-of-Experts Boosted Visual Perception-Driven and Semantic-Aware Quality Assessment for AI-Generated Images](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Yang_MoE-AGIQA_Mixture-of-Experts_Boosted_Visual_Perception-Driven_and_Semantic-Aware_Quality_Assessment_for_CVPRW_2024_paper.pdf) |          [GitHub](https://github.com/37s/MoE-AGIQA)          |
|                       AMFF-Net                       | [Adaptive Mixed-Scale Feature Fusion Network for Blind AI-Generated Image Quality Assessment](https://arxiv.org/pdf/2404.15163) |       [GitHub](https://github.com/TanSongBai/AMFF-Net)       |
|                         PSCR                         | [PSCR: Patches Sampling-based Contrastive Regression for AIGC Image Quality Assessment](https://arxiv.org/pdf/2312.05897) |         [GitHub ](https://github.com/jiquan123/PSCR)         |
|                         TIER                         | [TIER: Text-Image Encoder-based Regression for AIGC Image Quality Assessment](https://arxiv.org/pdf/2401.03854) | [GitHub](https://github.com/jiquan123/TIER?utm_source=catalyzex.com) |
|                         IPCE                         | [AIGC Image Quality Assessment via Image-Prompt Correspondence](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Peng_AIGC_Image_Quality_Assessment_via_Image-Prompt_Correspondence_CVPRW_2024_paper.pdf) |           [GitHub](https://github.com/pf0607/IPCE)           |
|                      RISEBench                       | [Envisioning Beyond the Pixels: Benchmarking Reasoning-Informed Visual Editing](https://arxiv.org/pdf/2504.02826) |      [GitHub](https://github.com/PhoenixZ810/RISEBench)      |
|                         GoT                          | [GoT: Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing](https://arxiv.org/pdf/2503.10639) |        [GitHub ](https://github.com/rongyaofang/GoT)         |
|                      SmartEdit                       | [SmartEdit: Exploring Complex Instruction-based Image Editing with Multimodal Large Language Models](https://arxiv.org/pdf/2312.06739) |      [GitHub ](https://yuzhou914.github.io/SmartEdit/)       |
|                         WISE                         | [WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation](https://arxiv.org/pdf/2503.07265) |       [GitHub](https://github.com/PKU-YuanGroup/WISE)        |
|                      KRIS-Bench                      | [KRIS-Bench: Benchmarking Next-Level Intelligent Image Editing Models](https://arxiv.org/pdf/2505.16707) | [GitHub](https://yongliang-wu.github.io/kris_bench_project_page/?utm_source=catalyzex.com) |
|                     CoT-editing                      | [Enhancing Image Editing with Chain-of-Thought Reasoning and Multimodal Large Language Models](https://ieeexplore.ieee.org/document/10890562) |                       *Not available*                        |

### Video

|                   Benchmark                   |                            Paper                             |                         Project Page                         |
| :-------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                    LMM-VQA                    | [LMM-VQA: Advancing Video Quality Assessment with Large Multimodal Models](https://arxiv.org/pdf/2408.14008) |          [GitHub](https://github.com/Sueqk/LMM-VQA)          |
|                    FineVQ                     | [FineVQ: Fine-Grained User Generated Content Video Quality Assessment](https://arxiv.org/pdf/2412.19238) |        [GitHub](https://github.com/IntMeGroup/FineVQ)        |
|                     VQA^2                     | [VQA2 : Visual Question Answering for Video Quality Assessment](https://arxiv.org/pdf/2411.03795) | [GitHub](https://github.com/Q-Future/Visual-Question-Answering-for-Video-Quality-Assessment) |
|                   Omni-VQA                    | [Scaling-up Perceptual Video Quality Assessment](https://arxiv.org/pdf/2505.22543) |                       *Not available*                        |
|                   LMM-PVQA                    | [Breaking Annotation Barriers: Generalized Video Quality Assessment via Ranking-based Self-Supervision](https://arxiv.org/pdf/2505.03631) |         [GitHub](https://github.com/clh124/LMM-PVQA)         |
|            Compare2Score paradigm             | [Adaptive Image Quality Assessment via Teaching Large Multimodal Model to Compare](https://arxiv.org/pdf/2405.19298) |          [GitHub](https://compare2score.github.io/)          |
|                  VQ-Insight                   | [VQ-Insight: Teaching VLMs for AI-Generated Video Quality Understanding via Progressive Visual Reinforcement Learning](https://arxiv.org/pdf/2506.18564) |       [GitHub](https://github.com/bytedance/Q-Insight)       |
|            Who is a Better Talker             | [Who is a Better Talker: Subjective and Objective Quality Assessment for AI-Generated Talking Heads](https://arxiv.org/pdf/2507.23343) |         [GitHub](https://github.com/zyj-2000/Talker)         |
|                     THQA                      | [THQA: A Perceptual Quality Assessment Database for Talking Heads](https://arxiv.org/pdf/2404.09003) |          [GitHub](https://github.com/zyj-2000/THQA)          |
|           Who is a Better Imitator            | [Who is a Better Imitator: Subjective and Objective Quality Assessment of Animated Humans](https://jhc.sjtu.edu.cn/~xiaohongliu/papers/2025imitator.pdf) |        [GitHub](https://github.com/zyj-2000/Imitator)        |
|                     MI3S                      | [MI3S: A multimodal large language model assisted quality assessment framework for AI-generated talking heads](https://pdf.sciencedirectassets.com/271647/1-s2.0-S0306457325X00054/1-s2.0-S0306457325002626/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEBcaCXVzLWVhc3QtMSJGMEQCICpU%2B72AWLLqhVTeZBT8KKIM6HqrxaWNhfx1XmMJws%2B5AiBz7ugtVWdvZs%2BTHix%2FkTyitgCHdYQs%2FgWZkhMsnEml%2FCqzBQhvEAUaDDA1OTAwMzU0Njg2NSIMXxdBi9ZA2gW0dKiMKpAF5OoKlRvyeBg%2Fzn2oPBKZoQjz4CSdu3Jy1XEkDSOfGZY3YD2JqyP7DAZMcltmpirR0C1tKf%2FnBi%2BvnHqdFbGm4YLBI9QNVmCRAKmMluCSHcLxBJbjYAazVMyJBPdWoFIQ2GbbQ42iOYdtNW%2FgtiyKU6MWcbb4KF8wKtLp6r4rj%2B38IM2IMlisN0277DwRBkYpjjCJk7G%2F4egUGIVY8Nje%2FGXlk0n1iASjUZKqsW7nQ759wgjmv5qgAm%2BwGqs7sB%2Bslv4GgLrXJzAeSqofQmfID%2BHHf%2Bz0B2aXMM3BUB3pOCW6szSEpHiDyk5izmFCYHxO4H5zYvySdzZvp9MOqnSYnWo2jDHIMHpfByhN7HWaZ223cZpAhQysTajsje1XNJGNatnUY5xFugh7B9zXpg9uuRhnkbUJN9%2BpFdjGoMHNxIFJUk5lZpX9xZ89jpFNzORGRw%2BuTxevGlob4jebb2j%2FMvlqOs7Jgfn5iZIcReWv7slfkHwgdcnnejkbFCPn5XMMgzku2%2FAUJOSmyZ6dV8xLxf%2Bn9a57n6VOhMORtBhMm7dx6I7fMAKSoknbWsnzl2nKql%2BwJp3cQlOsbgNEIVYKOtKft8cMsWDZt9w%2BLg6oLrEiqI3zEfsp4ccLQH0CfA7h9fAKQrZGySo%2FBsc%2FPmbw5A3yMJHHdLzFMBMXcuHqIxBeKPojYSH2ygYfuCwYBYbWe3IgbI21rjTYxTQ%2F1ACzX609a49d7OLKGp1HyDBQ137q2bk9DqFQAf27RDQojF3nid7U1gbOladQp9YR4aAQkNFj6emt4O39wTh2ug1%2F%2FBwURYkuuK7vxBxlumkJbDaNLV07ukn2tteddb51wYG%2F6Zo4czXQgItliQ6xcaQvzzsw9p61xQY6sgEo7PmFzMIHdYAjPhZXODMcw3XdsBC9Tl4%2BhQKHzQN6dIH1z6jx%2BJsEcp%2FTBp4H8Bis%2FuSSdKlnnRo5o7s2mR1q%2F0S5Df47BVAgtslpQIO8wRJ0lL1H5PZ8siXzsqHn%2BLi6xvqMH3q5vkE2NrHMWNTAwcEG7aUZ07cX9hyJUMqRLQSCS%2BPYnDw3NZSKZZFsdQbl7KfGHB%2F%2FBfKR%2FRtxOvBnzMcffup0Oq%2FOMspmpcIGwr%2F8&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250826T070451Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYXD6JI5HF%2F20250826%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=377df3c1bc014b91a05786f7ec5553e07819d6123bb58869433e31e33b0abda7&hash=5d3d2a43208117496897344bb132d068845e4d3a80667eb15d956e3915ac3678&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0306457325002626&tid=spdf-6c54bd1c-a6c5-4c5c-8126-5120be841082&sid=6bff0ae6832fe04476180138b052652ce1b0gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0e055b53545152065d&rr=97517a361b26e2f0&cc=hk&kca=eyJrZXkiOiJnV2RjazFEZldLb3lOVGUzdzVoSXdiQ2E1K05SNXVFemFqak5qK3lXVXoza0hmTExBTkxVMkp5bEVKcVZidEpEWFdGMUlDOTlQUHpoR095N2FzTnFQU0hOSi9CcndGL2NzVFNRSk5mcThQSG4vcm1YUEVwc1JrTWJlRFFMRmczQmd1Y0FWSGZ6UlRicktzaTZCZGpCcEt5OFQ1a2NHTHNTeG1UY3Y3TUNaWHl3ZkcxL0NRPT0iLCJpdiI6IjBhNmY1Mzc1YmFhZGEyNzA3ZDgwZGJiMGYwMmNmZTI3In0=_1756191904186) |                       *Not available*                        |
| An Implementation of Multimodal Fusion System | [An Implementation of Multimodal Fusion System for Intelligent Digital Human Generation](https://arxiv.org/pdf/2310.20251) |  [GitHub](https://github.com/zyj-2000/CUMT_2D_PhotoSpeaker)  |

### Audio

|               Benchmark               |                            Paper                             |                         Project Page                         |
| :-----------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                MOSNet                 | [MOSNet: Deep Learning-based Objective Assessment for Voice Conversion](https://arxiv.org/pdf/1904.08352) | [GitHub](https://github.com/lochenchou/MOSNet?utm_source=catalyzex.com) |
|               MOSA-Net+               | [A Study on Incorporating Whisper for Robust Speech Assessment](https://arxiv.org/pdf/2309.12766) |   [Hugging Face](https://huggingface.co/papers/2309.12766)   |
|               MOSLight                | [MOSLight: A Lightweight Data-Efficient System for Non-Intrusive Speech Quality Assessment](https://www.isca-archive.org/interspeech_2023/li23c_interspeech.pdf) |                       *Not available*                        |
|                 MBNet                 | [MBNet: MOS Prediction for Synthesized Speech with Mean-Bias Network](https://arxiv.org/pdf/2103.00110) | [GitHub](https://github.com/CalayZhou/MBNet?tab=readme-ov-file) |
|                DeePMOS                | [DeePMOS: Deep Posterior Mean-Opinion-Score of Speech](https://www.researchgate.net/profile/Fredrik-Cumlin/publication/373249119_DeePMOS_Deep_Posterior_Mean-Opinion-Score_of_Speech/links/676e7db6c1b0135465f772f5/DeePMOS-Deep-Posterior-Mean-Opinion-Score-of-Speech.pdf) |       [GitHub](https://github.com/Hope-Liang/DeePMOS)        |
|                 LDNet                 | [LDNet: Unified Listener Dependent Modeling in MOS Prediction for Synthetic Speech](https://arxiv.org/pdf/2110.09103) |         [GitHub](https://github.com/unilight/LDNet)          |
|                ADTMOS                 | [ADTMOS – Synthesized Speech Quality Assessment Based on Audio Distortion Tokens](https://ieeexplore.ieee.org/document/10938851) |       [GitHub](https://github.com/redifinition/ADTMOS)       |
|                 UAMOS                 | [Uncertainty-Aware Mean Opinion Score Prediction](https://arxiv.org/pdf/2408.12829) |                       *Not available*                        |
|          Audiobox Aesthetics          | [Meta Audiobox Aesthetics: Unified Automatic Quality Assessment for Speech, Music, and Sound](https://arxiv.org/pdf/2502.05139) | [GitHub](https://github.com/facebookresearch/audiobox-aesthetics) |
|              HighRateMOS              | [HighRateMOS: Sampling-Rate Aware Modeling for Speech Quality Assessment](https://arxiv.org/pdf/2506.21951) |                       *Not available*                        |
|            ALLMs-as-Judges            | [Audio-Aware Large Language Models as Judges for Speaking Styles](https://arxiv.org/pdf/2506.05984) |   [Hugging Face](https://huggingface.co/papers/2506.05984)   |
|                SALMONN                | [SALMONN: Towards Generic Hearing Abilities for Large Language Models](https://arxiv.org/pdf/2310.13289) |        [GitHub](https://github.com/bytedance/SALMONN)        |
|              Qwen-Audio               | [Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models](https://arxiv.org/pdf/2311.07919) |        [GitHub](https://github.com/QwenLM/Qwen-Audio)        |
|             Qwen2- Audio              | [Qwen2-Audio Technical Report](https://arxiv.org/pdf/2407.10759) |       [GitHub](https://github.com/QwenLM/Qwen2-Audio)        |
| natural language quality descriptions | [Enabling Auditory Large Language Models for Automatic Speech Quality Evaluation](https://arxiv.org/pdf/2409.16644) |        [GitHub](https://github.com/bytedance/SALMONN)        |
|              QualiSpeech              | [QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning and Descriptions](https://arxiv.org/pdf/2503.20290) | [Hugging Face](https://huggingface.co/datasets/tsinghua-ee/QualiSpeech) |
|             DiscreteEval              | [Evaluating Text-to-Speech Synthesis from a Large Discrete Token-based Speech Language Model](https://arxiv.org/pdf/2405.09768) |     [GitHub](https://swatsw.github.io/lrec24_eval_slm/)      |
|           EmergentTTS-Eval            | [EmergentTTS-Eval: Evaluating TTS Models on Complex Prosodic, Expressiveness, and Linguistic Challenges Using Model-as-a-Judge](https://arxiv.org/pdf/2505.23009) | [GitHub](https://github.com/boson-ai/EmergentTTS-Eval-public) |
|            InstructTTSEval            | [INSTRUCTTTSEVAL: Benchmarking Complex Natural-Language Instruction Following in Text-to-Speech Systems](https://arxiv.org/pdf/2506.16381) |  [GitHub](https://github.com/KexinHUANG19/InstructTTSEval)   |
|               Mos-Bench               | [MOS-Bench: Benchmarking Generalization Abilities of Subjective Speech Quality Assessment Models](https://arxiv.org/pdf/2411.03715) |   [Hugging Face](https://huggingface.co/papers/2411.03715)   |

### 3D

|    Benchmark     |                            Paper                             |                        Project Page                         |
| :--------------: | :----------------------------------------------------------: | :---------------------------------------------------------: |
|     NR-3DQA      | [No-Reference Quality Assessment for 3D Colored Point Cloud and Mesh Models](https://arxiv.org/pdf/2107.02041) |        [GitHub](https://github.com/zzc-1998/NR-3DQA)        |
|     MM-PCQA      | [MM-PCQA: Multi-Modal Learning for No-reference Point Cloud Quality Assessment](https://arxiv.org/pdf/2209.00244) |        [GitHub](https://github.com/zzc-1998/MM-PCQA)        |
|   GT23D-Bench    | [GT23D-Bench: A Comprehensive General Text-to-3D Generation Benchmark](https://arxiv.org/pdf/2412.09997) |          [GitHub](https://gt23d-bench.github.io/)           |
|     NeRF-NQA     | [NeRF-NQA: No-Reference Quality Assessment for Scenes Generated by NeRF and Neural View Synthesis Methods](https://arxiv.org/pdf/2412.08029) |      [GitHub](https://github.com/VincentQQu/NeRF-NQA)       |
| Explicit-NeRF-QA | [Explicit-NeRF-QA: A Quality Assessment Database for Explicit NeRF Model Compression](https://arxiv.org/pdf/2407.08165) |   [GitHub](https://github.com/YukeXing/Explicit-NeRF-QA)    |
|     NeRF-QA      | [NeRF-QA: Neural Radiance Fields Quality Assessment Database](https://arxiv.org/pdf/2305.03176) | [GitHub](https://github.com/pedrogcmartin/NeRF-QA-Database) |
|      NVS-QA      | [NeRF View Synthesis: Subjective Quality Assessment and Objective Metrics Evaluation](https://arxiv.org/pdf/2405.20078) |      [GitHub](https://github.com/pedrogcmartin/NVS-QA)      |
|       GSQA       | [GS-QA: Comprehensive Quality Assessment Benchmark for Gaussian Splatting View Synthesis](https://arxiv.org/pdf/2502.13196) |      [GitHub](https://github.com/pedrogcmartin/GS-QA)       |
| GPT-4V Evaluator | [GPT-4V(ision) is a Human-Aligned Evaluator for Text-to-3D Generation](https://arxiv.org/pdf/2401.04092) |           [GitHub](https://gpteval3d.github.io/)            |
|      Eval3D      | [Eval3D: Interpretable and Fine-grained Evaluation for 3D Generation](https://arxiv.org/pdf/2504.18509) |             [GitHub](https://eval3d.github.io/)             |
|   3DGen-Bench    | [3DGen-Bench: Comprehensive Benchmark Suite for 3D Generative Models](https://arxiv.org/pdf/2503.21745) |       [GitHub](https://zyh482.github.io/3DGen-Bench/)       |
|     LMM-PCQA     | [LMM-PCQA: Assisting Point Cloud Quality Assessment with LMM](https://arxiv.org/pdf/2404.18203?) |       [GitHub](https://github.com/Q-Future/LMM-PCQA)        |

## Leaderboards and Tools

|      Platform / Benchmark       |                            Paper                             |                         Project Page                         |
| :-----------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|            LMMs-Eval            | [LMMs-Eval: Reality Check on the Evaluation of Large Multimodal Models](https://arxiv.org/pdf/2407.12772) |   [GitHub](https://github.com/EvolvingLMMs-Lab/lmms-eval)    |
|           GenAI-Arena           | [Genai arena: An open evaluation platform for generative models](https://arxiv.org/pdf/2406.04485) | [Hugging Face](https://huggingface.co/spaces/TIGER-Lab/GenAI-Arena) |
|           OpenCompass           |                                                              |    [GitHub](https://github.com/open-compass/opencompass)     |
|   Epoch AI’s Benchmarking Hub   |                                                              |            [Website](https://epoch.ai/benchmarks)            |
|       Artificial Analysis       |                                                              |          [Website](https://artificialanalysis.ai/)           |
|    Scale’s SEAL Leaderboards    |                                                              |           [Website](https://scale.com/leaderboard)           |
|            FlagEval             | [FlagEvalMM: A Flexible Framework for Comprehensive Multimodal  Model Evaluation](https://arxiv.org/pdf/2506.09081) |        [Website](https://flageval.baai.ac.cn/#/home)         |
|            AGI-Eval             | [AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models](https://arxiv.org/pdf/2304.06364) |     [Website](https://agi-eval.cn/mvp/listSummaryIndex)      |
|              ReLE               |                                                              |  [Website](https://nonelinear.com/static/benchmarking.html)  |
| VLMEvalKit、OpenVLM Leaderboard | [VLMEvalKit: An Open-Source Toolkit for Evaluating Large Multi-Modality Models](https://arxiv.org/pdf/2407.11691) |     [GitHub](https://github.com/open-compass/VLMEvalKit)     |
|              HELM               | [Holistic Evaluation of Language Models](https://arxiv.org/pdf/2211.09110) | [Project Page](https://crfm.stanford.edu/helm/classic/v0.3.0/) |
|            LiveBench            | [LiveBench: A Challenging, Contamination-Limited LLM Benchmark](https://arxiv.org/pdf/2406.19314) |           [Project Page](https://livebench.ai/#/)            |
|            SuperCLUE            | [SuperCLUE: A Comprehensive Chinese Large Language Model Benchmark](https://arxiv.org/pdf/2307.15020) |          [Website](https://www.cluebenchmarks.com/)          |
|             AIBench             | [AIbench: Towards Trustworthy Evaluation Under the 45 Law](https://www.researchgate.net/profile/Zicheng-Zhang-9/publication/393362210_AIBENCH_TOWARDS_TRUSTWORTHY_EVALUA-_TION_UNDER_THE_45LAW/links/6867747be4632b045dc9b47c/AIBENCH-TOWARDS-TRUSTWORTHY-EVALUA-TION-UNDER-THE-45LAW.pdf) |                 [Website](https://aiben.ch)                  |


