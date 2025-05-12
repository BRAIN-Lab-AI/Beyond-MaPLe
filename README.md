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
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.
- **Problem 3:** There is limited capability in performing style transfer and generating diverse artistic variations.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.



