# Beyond MaPLe: Enhancing Multimodal Prompt Learning for Vision-Language Tasks

Below is a template for another sample project. Please follow this template.
# [Deep Learning Project Template] Enhanced Stable Diffusion: A Deep Learning Approach for Artistic Image Generation

## Introduction

Multimodal prompt learning has emerged as a powerful technique to enhance vision-language models, enabling improved generalization in zero-shot and few-shot learning tasks. In this project, we build upon the MaPLe framework (Multimodal Prompt Learning) by introducing three key enhancements:

1. **Prompt Dropout** – to regularize shallow prompts and reduce overfitting.
2. **Smarter Prompt Initialization** – using average CLIP text embeddings of class names for semantic anchoring.
3. **Selective Prompt Injection** – limiting prompt tokens to key transformer layers (1, 6, and 11) for better efficiency.

Our approach, dubbed **Beyond MaPLe**, is implemented and evaluated across 11 public datasets (excluding ImageNet and StanfordCars due to size constraints). Experiments demonstrate consistent improvement in novel class accuracy while preserving or slightly improving harmonic mean scores.

This project was completed as part of the ICS590 Deep Learning course at KFUPM.

## Project Metadata
### Authors
- **Team:** Omar Alandijani -- 201934090
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

### Project Documents
- **Report:** [BeyondMaPLe.pdf](https://github.com/BRAIN-Lab-AI/Beyond-MaPLe/blob/main/Beyond%20MaPLe.pdf)

### Reference Paper
- [MaPLe: Multi-modal Prompt Learning](https://arxiv.org/pdf/2210.03117)
- [Github Repository](https://github.com/muzairkhattak/multimodal-prompt-learning.git)

### Reference Datasets
- [DATASETS](https://github.com/muzairkhattak/multimodal-prompt-learning/blob/main/docs/DATASETS.md)


## Project Technicalities

### Terminologies
- **Vision-Language Model (VLM):** A deep learning model that jointly understands images and text by aligning them in a shared feature space.
- **CLIP (Contrastive Language–Image Pretraining):** A foundation model trained on image-text pairs using contrastive loss, enabling zero-shot classification.
- **Prompt Learning:** A technique where learnable tokens are prepended to inputs (text or image) to guide model behavior without updating the entire backbone.
- **Multimodal Prompting:** Applying prompt learning in both the text and image branches of a vision-language model (e.g., MaPLe).
- **Shallow vs. Deep Prompts:** Shallow prompts are applied to the input layer; deep prompts are inserted at intermediate transformer layers to influence multi-level representations.
- **Prompt Dropout:** A regularization technique that applies dropout to learned prompt embeddings during training, helping prevent overfitting.
- **Prompt Initialization:** The strategy used to initialize prompt tokens (random, repeated, or semantically derived).
- **Selective Injection:** Injecting prompts at chosen layers (e.g., layers 1, 6, 11) of the transformer to control the depth of influence and efficiency.
- **Base-to-Novel Generalization:** The model's ability to transfer knowledge from seen (base) classes to unseen (novel) classes during testing.
- **Zero-Shot / Few-Shot Learning:** Evaluation settings where the model has never (or barely) seen examples from the target classes.

### Problem Statements
- **Problem 1:** Prompt-based vision-language models like CLIP rely on static, hand-crafted prompts, which limit performance in domain-specific or novel class settings.
- **Problem 2:** Even with learnable prompts (as in CoOp or MaPLe), generalization to unseen classes often lags behind performance on base classes.
- **Problem 3:** Uniform injection of prompt tokens across all transformer layers introduces unnecessary complexity and redundancy without guaranteed gains.

### Loopholes or Research Areas
- **Prompt Generalization:** How can we better transfer prompt knowledge from base to novel classes without overfitting?
- **Semantic Anchoring:** Can we initialize prompts in a more meaningful way to align with actual class semantics?
- **Injection Efficiency:** Are all transformer layers equally important for prompting, or can we selectively choose optimal ones?

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Prompt Dropout:** Introduce dropout during training on prompt embeddings to reduce overfitting on base classes and improve robustness.
2. **Smarter Prompt Initialization:** Replace random or repeated initialization with averaged CLIP text embeddings of class names to semantically ground the prompts.
3. **Selective Prompt Injection:** Instead of prompting all layers, only inject prompts at layers 1, 6, and 11 — balancing representational depth and efficiency.

### Proposed Solution: Code-Based Implementation

This repository builds upon the official MaPLe framework for multimodal prompt learning. Our implementation focuses on lightweight, modular enhancements to prompt injection, initialization, and regularization.

The core solution includes:
- **Prompt Dropout:** Introduced during training to reduce overfitting on base classes and encourage better generalization.
- **Smarter Prompt Initialization:** Prompt tokens are initialized using the average CLIP embedding of class names, ensuring better semantic grounding.
- **Selective Prompt Injection:** Prompt tokens are injected only at selected transformer layers (1, 6, 11) to optimize representational depth and computational efficiency.

### 🔧 Key Components

- **`maple.py`**: A self-contained file that implements the proposed enhancements to the MaPLe framework. This script modifies prompt dropout, initialization, and layer injection logic.

The rest of the implementation relies on the [official MaPLe GitHub repository](https://github.com/muzairkhattak/multimodal-prompt-learning.git), which provides the full backbone, dataset utilities, and training pipeline.

Refer to the following files in the original MaPLe repo for complete usage:
- 📄 **`INSTALL.md`** – Setup instructions and dependencies.  
- 📄 **`DATASETS.md`** – Dataset preparation and folder structure.  
- 📄 **`RUN.md`** – How to train and evaluate the model using provided configs.

## Model Workflow

The Beyond MaPLe framework builds upon the original MaPLe architecture to improve base-to-novel generalization in vision-language models. The workflow can be broken down into three main phases:

1. **Input Encoding:**
   - **Text Prompts:** Class names are used as textual prompts (e.g., `"a photo of a tiger"`). These are embedded using CLIP’s text encoder.
   - **Image Input:** Images are passed through CLIP’s vision transformer for feature extraction.
   - **Prompt Initialization:** Learnable prompt tokens are initialized using the average CLIP embedding of class names, offering semantic grounding from the start.

2. **Prompt Integration and Injection:**
   -  **Shallow Prompts:** Token sequences are prepended to both the image and text inputs before passing them into the transformer blocks.
   - **Selective Layer Injection:** Instead of injecting prompts into all transformer layers, Beyond MaPLe selectively injects at layers 1, 6, and 11 for efficiency and deeper semantic influence.
   - **Prompt Dropout:** During training, dropout is applied to prompt tokens to prevent overfitting on base classes and encourage generalization to novel classes.

3. **Feature Alignment and Classification:**
   - **Contrastive Learning:** CLIP’s contrastive loss is used to align visual and textual representations in a shared embedding space.
   - **Zero-/Few-Shot Evaluation:** At inference time, the model classifies unseen images using only class name prompts — no additional training is needed for novel classes.
   - **Prediction:** The similarity between image and text embeddings determines the predicted class in a zero-shot fashion.


## How to Run the Code

This project builds on top of the official [MaPLe repository](https://github.com/muzairkhattak/multimodal-prompt-learning.git), with enhancements implemented in a modified version of the `maple.py` file.

Follow these steps to integrate and run the improved Beyond MaPLe version:

### 🔧 Steps:

1. **Clone the official MaPLe repository:**
    
    ```bash
    git clone https://github.com/muzairkhattak/maple.git
    cd maple
    ```

2. **Set up the environment:**
    
    Follow the installation guide in the original `INSTALL.md` file:
    
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the datasets:**
    
    Refer to the `DATASETS.md` file in the MaPLe repository to download and organize all supported datasets correctly.

4. **Replace the trainer file with our enhanced version:**
    
    Copy the `maple.py` file from this repository and overwrite the original one inside the MaPLe repo's `trainers/` folder:
    
    ```bash
    cp path/to/this_repo/maple.py path/to/maple/trainers/maple.py
    ```

5. **Train and evaluate the model:**
    
    Use any of the provided configuration files in the original repo or create your own. Example:
    
    ```bash
    python train.py --config-file configs/oxford_pets/maple.yaml
    ```

    You can find more details on training and evaluation in the official `RUN.md`.

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.



