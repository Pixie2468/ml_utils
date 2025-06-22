# 🧠 ml_utils: Modular Machine Learning Utilities for PyTorch

`ml_utils` is a modular, extensible training framework built on PyTorch, designed to streamline model training across multiple domains. Currently focused on **vision tasks** using `torchvision`, this repository is being developed with a **plugin architecture** in mind to support future domains like `torchaudio`, `torchrec`, and `torchtext`.

> 💡 Ideal for researchers, ML engineers, and practitioners who want a clean, maintainable PyTorch training loop that supports rapid experimentation and domain scalability.

---

## 📦 Features

- ✅ **Clean Training Loop**: Simple, extensible engine built with PyTorch
- 🖼️ **Vision Support**: Out-of-the-box training for common `torchvision` models
- 🧠 **Custom Model Support**: Easily integrate your own `nn.Module` architectures
- 📊 **Training Visualization**: Automatically plot accuracy/loss curves
- 💾 **Model Saving & Checkpointing**: Save best-performing models and reload them
- 🔌 **Plugin System (Planned)**: Add support for other domains (audio, text, recsys) with minimal changes to the core
- 🧪 **Reproducible Training**: Seed control, logging, and metrics saving

---

## 📁 Directory Structure

```
ml_utils/
├── ml_utils/
│   ├── engine.py         # Training loop and evaluation logic
│   ├── model.py          # torchvision + custom model definitions
│   ├── utils.py          # Utility functions (device, saving, plotting)
│   ├── data\_setup\_image.py # Vision-specific dataloaders
│   └── cli.py (planned)  # Plugin-enabled command-line interface
│
├── train.py              # Main script for training
├── requirements.txt
├── README.md
└── examples/
    ├── example_usage.py
```

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Pixie2468/ml_utils.git
cd ml_utils
pip install -r requirements.txt
```

---

## ⚙️ Basic Usage

### 🧾 List Available Models

```bash
python train.py --list-models
```

---

### 🏁 Train a Pretrained Vision Model (e.g., ResNet18 on CIFAR10)

```bash
python train.py \
  --model resnet18 \
  --pretrained \
  --dataset cifar10 \
  --data-path ./data \
  --batch-size 64 \
  --epochs 20 \
  --lr 0.01 \
  --optimizer sgd \
  --output-dir ./outputs/resnet18_cifar10 \
  --save-best \
  --plot-curves
```

---

### 🧱 Train a Custom Model

```bash
python train.py \
  --model custom_cnn \
  --dataset cifar10 \
  --data-path ./data \
  --batch-size 32 \
  --epochs 15 \
  --lr 0.001 \
  --optimizer adam \
  --dropout 0.3 \
  --output-dir ./outputs/custom_cnn
```

---

## 📂 Use a Custom Dataset

Dataset must follow this format (compatible with `torchvision.datasets.ImageFolder`):

```
custom_dataset/
├── train/
│   ├── class1/
│   └── class2/
└── val/
    ├── class1/
    └── class2/
```

Command:

```bash
python train.py \
  --model resnet18 \
  --dataset custom \
  --data-path ./custom_dataset \
  --batch-size 32 \
  --epochs 10 \
  --output-dir ./outputs
```

---

## ➕ Add a New Dataset Class

Modify `train.py`:

```python
elif args.dataset.lower() == 'my_dataset':
    train_dataset = MyDataset(args.data_path, transform=transforms_dict['train'])
    val_dataset = MyDataset(args.data_path, train=False, transform=transforms_dict['val'])
    num_classes = train_dataset.num_classes
```

---

## 🔌 Plugin Architecture (Planned)

To keep the codebase modular and scalable, `ml_utils` is being built with a plugin system:

### Future Plugins

| Domain              | Library     | Status      |
| ------------------- | ----------- | ----------- |
| Computer Vision     | torchvision | ✅ Supported |
| Audio               | torchaudio  | 🔜 Planned   |
| NLP/Text            | torchtext   | 🔜 Planned   |
| Recommender Systems | torchrec    | 🔜 Planned   |

Each plugin will handle its own:

* Dataset loader
* Model loader
* Preprocessing pipeline

This avoids polluting the main training logic with domain-specific code.

---

## 🧪 Example Script

```bash
cd examples
python example_usage.py
```

---

## ⚠️ Notes

* ✅ **Cross-platform**: Works on Linux, macOS, and Windows
* ⚠️ **Windows**: Dataloader workers are capped at 4 to avoid multiprocessing issues
* ❗ **Class Matching**: Training and validation folders must contain the same class names

---

## 🧠 Contributing

We welcome PRs and feature additions! Here's how you can contribute:

### Add a New Model

```python
class MyModel(nn.Module):
    ...

CUSTOM_MODELS['my_model'] = MyModel
```

Then run:

```bash
python train.py --model my_model
```

### Add a Plugin (future)

In the upcoming plugin system:

```python
from ml_utils.core.registry import register_plugin

@register_plugin("audio")
class AudioPlugin:
    def get_dataloaders(self, args): ...
    def get_model(self, args): ...
```

---

## 🔗 Related Projects

* [torchvision](https://pytorch.org/vision/stable/index.html)
* [torchaudio](https://pytorch.org/audio/stable/index.html)
* [torchrec](https://pytorch.org/torchrec/)
* [PyTorch Lightning](https://www.pytorchlightning.ai/) (if you want a more batteries-included approach)

---

## 🙋‍♀️ Questions or Feedback?

Open an [issue](https://github.com/Pixie2468/ml_utils/issues) or start a discussion. Let’s build a powerful, extensible ML training system together.
