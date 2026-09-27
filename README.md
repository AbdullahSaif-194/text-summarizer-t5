# 📝 Text Summarizer — Fine-tuned T5-Small

A **fine-tuned T5-small** transformer model for abstractive text summarization, trained on the **SAMSum** conversational dataset. Includes a Gradio web interface for easy interaction.

## 🚀 Features

- 🤗 Fine-tuned **T5-small** on SAMSum (dialogue summarization dataset)
- 📊 Trained with **Hugging Face Transformers** + Trainer API
- 🌐 Web UI built with **Gradio**
- 💾 Model saved in `safetensors` format

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Base Model | T5-small (Google) |
| Fine-tuning | Hugging Face Transformers |
| Dataset | SAMSum (not included) |
| UI | Gradio |
| Language | Python 3.12 |

## 📁 Project Structure

```
Text_Summarizer/
├── Text_summarizer.ipynb     # Training & fine-tuning notebook
├── text_summarizer.py        # Inference script with Gradio UI
├── index.html                # Web interface
└── Text_Summarizer_Model/    # Saved model weights (not tracked - too large)
```

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/AbdullahSaif-194/text-summarizer-t5.git
cd text-summarizer-t5

# 2. Install dependencies
pip install transformers datasets torch gradio

# 3. Run fine-tuning (or skip if using pre-trained)
jupyter notebook Text_summarizer.ipynb

# 4. Run the app
python text_summarizer.py
```

## 📊 Dataset

This project uses the **SAMSum** dataset (conversational summarization).
Dataset not included in repo — download it from [HuggingFace Datasets](https://huggingface.co/datasets/samsum):
```python
from datasets import load_dataset
dataset = load_dataset("samsum")
```

## 📈 Results

Model fine-tuned for dialogue/conversation summarization using ROUGE metrics for evaluation.
