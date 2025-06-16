# 🎓 LecEval: An Automated Metric for Evaluating Multimodal Educational Presentations

<div align="center">

[![Paper](https://img.shields.io/badge/📄-Paper-blue)](https://arxiv.org/abs/2505.02078v1) 
[![Dataset](https://img.shields.io/badge/🗂️-Dataset-green)](#dataset) 
[![Model](https://img.shields.io/badge/🤖-Model-orange)](https://huggingface.co/Joylimjy/LecEval) 
[![Evaluation](https://img.shields.io/badge/📊-Evaluation-purple)](#evaluation) 
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

[**🚀 Quick Start**](#quick-start) | [**📊 Leaderboard**](#leaderboard) | [**🛠️ Installation**](#installation) | [**🔧 Toolkit**](#toolkit) | [**📖 Documentation**](src/README.md)

</div>

## Overview

**LecEval** is a specialized automated metric designed for evaluating multimodal educational presentations, addressing the critical need for objective assessment tools in educational technology. Drawing inspiration from **Mayer's Cognitive Theory of Multimedia Learning**, which emphasizes the importance of aligning verbal and visual information in educational contexts, LecEval establishes four critical scoring rubrics to comprehensively assess presentation effectiveness.

### Key Features

- 🎯 **Domain-Specific**: Tailored specifically for educational presentation evaluation
- 🧠 **Theory-Grounded**: Based on established cognitive learning principles
- 🔄 **Multimodal**: Analyzes both visual slides and explanatory text
- 📏 **Multi-Dimensional**: Evaluates across four critical pedagogical rubrics
- 🤖 **Automated**: Provides consistent, scalable evaluation without human bias
- 🔗 **Aligned**: Achieves strong correlation with human expert assessments

### Evaluation Rubrics

| Rubric | Description | Focus Area |
|--------|-------------|------------|
| **Content Relevance (CR)** | Alignment between slide visuals and explanatory text | Information Coherence |
| **Expressive Clarity (EC)** | Clarity and comprehensibility of the presentation | Communication Effectiveness |
| **Logical Structure (LS)** | Organization and flow of information | Pedagogical Structure |
| **Audience Engagement (AE)** | Potential to capture and maintain learner attention | Learning Engagement |

<a name="quick-start"></a>
## 🚀 Quick Start

### Basic Usage

```python
from leceval import LecEvalMetric

# Initialize the metric
metric = LecEvalMetric()

# Evaluate a single presentation slide
slide_path = "path/to/slide.jpg"
explanation_text = "Today we'll explore the fundamentals of machine learning..."

scores = metric.evaluate(slide_path, explanation_text)
print(f"Content Relevance: {scores['content_relevance']:.2f}")
print(f"Expressive Clarity: {scores['expressive_clarity']:.2f}")
print(f"Logical Structure: {scores['logical_structure']:.2f}")
print(f"Audience Engagement: {scores['audience_engagement']:.2f}")
```

### Batch Evaluation

```python
# Evaluate multiple presentations
presentations = [
    {"slide": "slide1.jpg", "text": "Introduction to AI..."},
    {"slide": "slide2.jpg", "text": "Neural networks are..."},
]

results = metric.batch_evaluate(presentations)
```

<a name="installation"></a>
## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (for GPU acceleration)

```bash
git clone https://github.com/JoylimJY/LecEval.git
cd LecEval
pip install -e .
```


<a name="dataset"></a>
## 🗂️ Dataset

### Problem Formulation

Consider a presentation slide $S$ containing multimodal content, such as text, images, and diagrams. The evaluation problem can be formulated as assessing the quality of explanatory textual sequence $T$ that integrates slide content with external knowledge $\mathcal{K}=\Phi(S)$:

$$T_{S} = \xi(S,\mathcal{K})=\xi(S,\Phi(S))$$

where $\xi(\cdot)$ represents the function that produces coherent explanatory text, and $\Phi(\cdot)$ denotes auxiliary knowledge extraction.

Our evaluation metric is defined as:
$$\mathcal{A}=\mathcal{A}(S,T_S)=\mathcal{A}(S,\xi(S,\mathcal{K}))$$

which assesses educational presentations across content alignment and knowledge enhancement dimensions.

### Data Construction Pipeline

<div align="center">
  <img src="assets/dataset.png" alt="Dataset Construction Framework" width="95%"/>
</div>

Our data construction follows a rigorous three-stage process:

#### 1. **Heterogeneous Data Integration**
- Collection of online lecture videos across multiple domains
- Extraction of high-quality slide images and audio transcripts
- Preprocessing to ensure multimodal data quality

#### 2. **Multimodal Alignment and Refinement**
- Precise alignment between transcriptions and corresponding slides
- Refinement of raw transcriptions to enhance clarity and coherence
- Preservation of speaker's original pedagogical intent

#### 3. **Fine-grained Presentation Assessment**
- Engagement of experienced human annotators with educational backgrounds
- Systematic evaluation across four predefined rubrics
- Quality control through inter-annotator agreement analysis

### Dataset Statistics

Our curated **LecEval Dataset** comprises:

- **📚 56 lectures** across multiple educational domains
- **📄 2,097 slide-text pairs** with comprehensive annotations
- **👥 Multiple annotators** per sample ensuring reliability
- **🏷️ 4 evaluation rubrics** per sample

<div align="center">
  <img src="assets/dataset-stats.png" alt="Dataset Statistics" width="60%"/>
  <p><em>Dataset composition: outer ring shows lecture count, middle ring shows total hours, inner ring shows slide count</em></p>
</div>

### Data Format

Each sample follows this structure:

```json
{
  "id": "ml-1_10_slide_000", 
  "slide": "/data/images/ml-1/10/slide_000.png",
  "transcript": "Welcome, everyone, to Lecture 5.2 on Alignment and Representation. Today we'll explore how neural networks learn to align different modalities...",
  "rate": {
    "content_relevance": [5, 5, 4],    
    "expressive_clarity": [5, 4, 5],
    "logical_structure": [4, 5, 5],
    "audience_engagement": [3, 3, 4]
  }
}
```

<a name="model"></a>
## 🤖 Model Architecture

### Base Model

We employ **MiniCPM-Llama3-V2.5** (8B parameters) as our backbone model, chosen for its:
- Strong multimodal understanding capabilities
- Efficient parameter usage
- Robust performance across diverse domains

### Training Strategy

- **Supervised Fine-Tuning (SFT)** on our curated dataset
- **Multi-task Learning** across all four evaluation rubrics
- **Domain Adaptation** techniques for educational content

### Model Access

Access our trained models through our Hugging Face repository:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("Joylimjy/LecEval")
tokenizer = AutoTokenizer.from_pretrained("Joylimjy/LecEval")
```

### Human-Model Alignment

<div align="center">
  <img src="assets/leceval-human.png" alt="Human-Model Correlation" width="60%"/>
  <p><em>LecEval consistently assigns higher scores to presentations rated highly by human evaluators</em></p>
</div>

<a name="evaluation"></a>
## 📊 Evaluation Results

### Experimental Setup

We conduct comprehensive experiments comparing LecEval against existing metrics across two categories:

1. **Reference-based Metrics**: Traditional NLP metrics (BLEU, ROUGE)
2. **Prompt-based LLM Evaluators**: Modern LLM-based evaluation approaches

### Performance Comparison

**Spearman Correlation (ρ) with Human Evaluation:**

<table align="center">
  <thead>
    <tr>
      <th><strong>Metric</strong></th>
      <th><strong>Content<br>Relevance</strong></th>
      <th><strong>Expressive<br>Clarity</strong></th>
      <th><strong>Logical<br>Structure</strong></th>
      <th><strong>Audience<br>Engagement</strong></th>
      <th><strong>Overall</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>BLEU-4</td>
      <td>0.12</td>
      <td>0.10</td>
      <td>0.11</td>
      <td>0.18</td>
      <td>0.13</td>
    </tr>
    <tr>
      <td>ROUGE-L</td>
      <td>0.24</td>
      <td>0.29</td>
      <td>0.15</td>
      <td>0.07</td>
      <td>0.19</td>
    </tr>
    <tr>
      <td>GPT-4V</td>
      <td>0.29</td>
      <td>0.20</td>
      <td>0.27</td>
      <td>0.32</td>
      <td>0.27</td>
    </tr>
    <tr>
      <td>G-Eval</td>
      <td>0.26</td>
      <td>-0.05</td>
      <td>0.13</td>
      <td>0.04</td>
      <td>0.09</td>
    </tr>
    <tr style="background-color: #f0f8ff;">
      <td><strong>LecEval</strong></td>
      <td><strong>0.65</strong></td>
      <td><strong>0.84</strong></td>
      <td><strong>0.80</strong></td>
      <td><strong>0.79</strong></td>
      <td><strong>0.77</strong></td>
    </tr>
  </tbody>
</table>

### Key Findings

- **📈 Superior Performance**: LecEval achieves 0.77 overall correlation vs. 0.27 for the best baseline
- **🎯 Consistent Excellence**: Strong performance across all four evaluation rubrics
- **🔍 Domain Specificity**: Significant improvement over general-purpose metrics
- **⚡ Efficiency**: Fast evaluation compared to LLM-based approaches

<a name="leaderboard"></a>
## 📊 Leaderboard

Submit your results to our leaderboard! We welcome comparisons with new methods.

| Rank | Method | Overall ρ | CR | EC | LS | AE | Paper |
|:----:|:-------|:---------:|:--:|:--:|:--:|:--:|:-----:|
| 1 | **LecEval** | **0.77** | **0.65** | **0.84** | **0.80** | **0.79** | [Ours] |
| 2 | GPT-4V | 0.27 | 0.29 | 0.20 | 0.27 | 0.32 | [OpenAI] |
| 3 | G-Eval | 0.09 | 0.26 | -0.05 | 0.13 | 0.04 | [Liu et al.] |

## Toolkit

The LecEval toolkit provides comprehensive tools for analyzing and evaluating multimodal educational presentations. For detailed information about the toolkit components and usage, please refer to our [Toolkit Documentation](src/README.md).

Key features include:

- **Dataset Handler**: Load and manage multimodal lecture data including slides, transcripts, and annotations
- **Analyzer**: Statistical analysis tools for lecture content and performance metrics
- **Evaluator**: Robust evaluation framework for model predictions
- **Visualizer**: Comprehensive visualization tools for data analysis and results

For detailed usage instructions and examples, check out the [Toolkit Documentation](src/README.md).

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

- 🐛 **Bug Reports**: Report issues via GitHub Issues
- 💡 **Feature Requests**: Suggest new features or improvements
- 📝 **Documentation**: Help improve our documentation
- 🔬 **Research**: Share your findings using LecEval
- 💻 **Code**: Submit pull requests with improvements

### Development Setup

```bash
git clone https://github.com/JoylimJY/LecEval.git
cd LecEval

# Create development environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## 📜 Citation

If you use LecEval in your research, please cite our paper:

```bibtex
@misc{joy2025leceval,
      title={LecEval: An Automated Metric for Multimodal Knowledge Acquisition in Multimedia Learning}, 
      author={Joy Lim Jia Yin and Daniel Zhang-Li and Jifan Yu and Haoxuan Li and Shangqing Tu and Yuanchun Wang and Zhiyuan Liu and Huiqin Liu and Lei Hou and Juanzi Li and Bin Xu},
      year={2025},
      eprint={2505.02078},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.02078}, 
}
```

## 🚀 News & Updates

- **[2025-06]** 🎉 Paper submitted to CIKM resource track!
- **[2025-05]** 📊 Dataset and model publicly released
- **[2024-10]** 🏆 Achieved state-of-the-art performance on educational presentation evaluation
- **[2024-10]** 🔬 Initial research findings published

## 🙏 Acknowledgments

- Thanks to all annotators who contributed to dataset creation
- Built on the foundations of Mayer's Cognitive Theory of Multimedia Learning
- Inspired by advances in multimodal AI and educational technology

## 📧 Contact

For questions, collaborations, or support:

- **Issues**: [GitHub Issues](https://github.com/JoylimJY/LecEval/issues)
- **Discussions**: [GitHub Discussions](https://github.com/JoylimJY/LecEval/discussions)
- **Email**: [lin-jy23@mails.tsinghua.edu.cn](lin-jy23@mails.tsinghua.edu.cn)

<div align="center">

**Made with ❤️ for the educational technology community**

[⭐ Star us on GitHub](https://github.com/JoylimJY/LecEval) | [💬 Join our Community](https://project.maic.chat)

</div>

