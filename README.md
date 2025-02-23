# 🕊️ DOVE: A Large-Scale Dataset for LLM Evaluation (v1.0)

Official repository for the DOVE dataset containing 300M predictions across various prompt variations, enabling systematic study of LLM sensitivity and meaningful evaluation.

<p align="center">
    <img 
        alt="GitHub" 
        src="https://cdn.jsdelivr.net/npm/simple-icons@v7/icons/github.svg" 
        width="20" 
        height="20" 
        style="vertical-align: middle; margin-right: 4px;"
    /> <a href="https://doveevaluation.github.io/" target="_blank">Code</a> |
    <img
        alt="arXiv"
        src="https://commons.wikimedia.org/wiki/File:ArXiv_logo_2022.svg"
        width="20"
        height="20"
        style="vertical-align: middle; margin-right: 4px; margin-left:4px;"
    />
    <a href="https://arxiv.org/abs/XXXX.XXXXX" target="_blank">Paper</a> |
   🤗 <a href="https://huggingface.co/datasets/DOVevaluation/Dove" target="_blank">Dataset</a> |
   📧 <a href="mailto:eliyahaba@gmail.com">Contact</a>

</p>



## Community Contributions Welcome! 🤝
We envision DOVE as a living, community-driven resource for LLM evaluation. We have two main paths for contribution:

### Share Your Data 🗃️
We welcome data contributions that align with DOVE's core principles - studying prompt variations and model behavior. Your data doesn't need to contain all the fields defined in our schema format (see Table 3 in the paper), but should maintain systematic evaluation principles by including model predictions with variations in at least one dimension.

Contributors who provide significant data will be invited to join as co-authors on the next version of both the paper and dataset.

### Suggest Future Directions 💡
We're excited to hear your ideas about expanding DOVE. Whether it's exploring new domains, adding evaluation dimensions, incorporating different models and tasks, or any other innovative approaches - your input will help shape the future of LLM evaluation.

To contribute data or make suggestions:
- Email: eliya.habba@mail.huji.ac.il
- Visit our project page for contribution guidelines
- Follow us on Hugging Face for updates

## About
DOVE provides a large-scale dataset for studying how Language Models (LLMs) respond to different ways of asking the same question, focusing on:
- 300M model predictions across various evaluation benchmarks
- Systematic variations across multiple prompt dimensions
- Insights into model sensitivity and evaluation methodology
- Efficient methods for prompt selection and evaluation

## Installation

```bash
# Clone the repository
git clone git@github.com:DOVevaluation/DOVE.git
cd DOVE

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Dataset Loading

```python
from datasets import load_dataset
from pathlib import Path

# Load specific model/language/shots combination
def load_dove_subset(model_name, language="en", shots=0):
   base_path = f"DOVevaluation/Dove-full/{model_name}/{language}/shots_{shots}"
   return load_dataset(base_path)

# Available models:
# - Llama-3.2-1B-Instruct
# - OLMoE-1B-7B-0924-Instruct
# - Meta-Llama-3-8B-Instruct 
# - Llama-3.2-3B-Instruct
# - Mistral-7B-Instruct-v0.3

# Examples
llama_en_zero = load_dove_subset("Llama-3.2-1B-Instruct", language="en", shots=0)
mistral_fr_five = load_dove_subset("Mistral-7B-Instruct-v0.3", language="fr", shots=5)
```

## Project Structure
```
model_name/
   └── language/
       └── shots_N/
           └── data files
```

## Hardware Requirements
- Storage requirements:
 - Full Version: 4TB
 - Lite Version: 200GB
- Processing capabilities scale with analysis needs

## Citation
```bibtex
@article{dove2024,
 title={DOVE: A Large-Scale Multi-Dimensional Predictions Dataset Towards Meaningful LLM Evaluation},
 author={Anonymous},
 journal={arXiv preprint arXiv:XXXX.XXXXX},
 year={2024}
}
```

## License
This dataset is licensed under the Computational Data License Agreement v2 (CDLAv2). For full license terms, see: https://cdla.dev/permissive-2.0/