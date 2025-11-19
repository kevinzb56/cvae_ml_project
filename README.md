# Conditional Variational Autoencoder for Face Attribute Editing

## Project Overview
| Original | Smiling Edit | Moustache Edit | Glasses Edit |
|:--------:|:------------:|:--------------:|:------------:|
| <img src="./results/output-Original.png" style="width:180px; height:180px; object-fit:cover;"/> | <img src="./results/output-smile.png" style="width:180px; height:180px; object-fit:cover;"/> | <img src="./results/output-moustache.png" style="width:180px; height:180px; object-fit:cover;"/> | <img src="./results/output-glasses.png" style="width:180px; height:180px; object-fit:cover;"/> |


This project implements a CVAE model capable of editing facial images by adding or modifying three key attributes:

- Eyeglasses
- Smiling
- Mustache

The model learns to generate realistic facial variations while conditioning on selected attributes, enabling controlled face synthesis and attribute transfer.

## System Architecture

### Conditional Variational Autoencoder (CVAE)

![Original](./results/cvae-arch.png)

The CVAE architecture consists of four main components:

**Encoder Network**

- Input concatenates RGB image with attribute channels (3 + 3 = 6 channels)
- Four convolutional layers with LeakyReLU activation
- Channel progression: 128 → 256 → 512 → 1024
- Stride-2 convolutions for spatial downsampling
- Output flattened to 1024 × 4 × 4 = 16,384 dimensions

**Latent Space**

- Continuous representation of facial features
- Dimension: 128 (configurable)
- Mean (μ) and log-variance (logvar) computed via fully connected layers
- Sampled from learned Gaussian distribution using reparameterization trick
- Conditioned by concatenating attribute vectors (z + c)

**Decoder Network**

- Fully connected layer expands latent + attribute vectors to 1024 × 4 × 4
- Four transposed convolutional layers with ReLU activation
- Channel progression: 1024 → 512 → 256 → 128 → 3
- Tanh activation for final layer (output range [-1, 1])
- Generates realistic 64 × 64 RGB facial images

**Loss Function**

- Reconstruction Loss (MSE): Ensures fidelity to input images
- KL Divergence: Regularizes the latent space distribution
- Total Loss: `L = MSE + β * KLD` where β = 4.0

### Key Features

- Attribute conditioning enables controlled generation
- Variational inference provides diverse outputs
- Spatially-conditioned attributes for better control
- End-to-end differentiable training pipeline

## Dataset

**CelebA (CelebFaces Attributes Dataset)**

- 202,599 face images (178 × 218 pixels original)
- 40 binary attribute labels per image
- High-quality celebrity face photographs
- Preprocessing: Center crop to 178 × 218, resize to 64 × 64

**Selected Attributes** (3 out of 41 available)

1. Eyeglasses - presence of eyewear
2. Smiling - smiling expression
3. Mustache - presence of facial hair

**Image Normalization**

- Mean: [0.5, 0.5, 0.5]
- Std Dev: [0.5, 0.5, 0.5]
- Normalized range: [-1, 1]

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- numpy, pandas, matplotlib
- Pillow
- tqdm
- kaggle (for dataset download)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/chaitra-samant/cvae-celeba-project.git
cd cvae-celeba-project
```

### 2. Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
.\venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 3. Configure Kaggle API

1. Visit https://kaggle.com/account
2. Scroll to API section
3. Click "Create New API Token" (downloads `kaggle.json`)
4. Place the file in the appropriate location:

```
macOS/Linux: ~/.kaggle/kaggle.json
Windows:     C:\Users\<Your-Username>\.kaggle\kaggle.json
```

Create the `.kaggle` directory if it does not exist.

### 4. Download CelebA Dataset

```bash
python download_data.py
```

This downloads and extracts the dataset (~1.3 GB).

## Usage

### Training the Model

The model can be trained using the Jupyter notebook or Python scripts:

```bash
python main.py
```

#### Custom Training Parameters

```bash
python main.py --epochs 75 --lr 5e-5 --batch-size 128 --latent-dim 128
```

**Available Arguments:**

- `--epochs`: Number of training epochs (default: 75)
- `--lr`: Learning rate (default: 5e-5)
- `--batch-size`: Batch size for training (default: 128)
- `--latent-dim`: Dimensionality of latent space (default: 128)
- `--beta`: KL divergence weight (default: 4.0)

### Model Output

- Generated samples: `samples_64/`
- Trained model checkpoint: `cvae_eyeglasses_smiling_mustache.pth`
- Training logs and metrics saved automatically

## Results

### Output Examples

The model generates facial variations by conditioning on specific attributes. For each sample, the model generates four variations using the same latent vector:

**Example 1: Expression Modification**

<table>
  <tr>
    <td><img src="samples_v3/epoch100_neutral.png" alt="Neutral" width="260"/></td>
    <td><img src="samples_v3/epoch100_smiling_only.png" alt="Smiling only" width="260"/></td>
  </tr>
  <tr>
    <td align="center">Neutral</td>
    <td align="center">Smiling Only</td>
  </tr>
</table>

<br>

**Example 2: Original v/s All 3 attributes**

<table>
  <tr>
    <td><img src="samples_v3/epoch100_neutral.png" alt="Neutral" width="260"/></td>
    <td><img src="samples_v3/epoch100_all_three.png" alt="All three attributes" width="260"/></td>
  </tr>
  <tr>
    <td align="center">Neutral</td>
    <td align="center">Eyeglasses + Smiling + Mustache</td>
  </tr>
</table>




### Training Metrics

- **Initial Learning Rate:** 5e-5 (Adam optimizer)
- **Total Epochs:** 75
- **Batch Size:** 128
- **Image Resolution:** 64 × 64 pixels
- **Total Training Samples:** 64 generated variations

## Project Structure

```
cvae-celeba-project/
├── main.py                          # Main training script
├── download_data.py                 # Dataset download utility
├── requirements.txt                 # Project dependencies
├── notebooks/
│   └── cvae-model.ipynb            # Complete training notebook
├── models/
│   └── cvae.py                      # CVAE architecture
├── data/
│   └── celeba_loader.py             # Data loading utilities
├── utils/
│   ├── training.py                  # Training loop functions
│   └── visualization.py             # Image visualization tools
├── samples_64/                      # Generated samples directory
├── results/                         # Output images and results
└── README.md                        # This file
```

## Training Details

**Model Configuration**

- Model: Conditional Variational Autoencoder
- Input Size: 64 × 64 RGB images
- Latent Dimension: 128
- Number of Attributes: 3
- Base Channels: 128

**Training Hyperparameters**

- Optimizer: Adam
- Learning Rate: 5e-5
- Batch Size: 128
- Number of Epochs: 75
- Beta (KL weight): 4.0
- Loss: Reconstruction (MSE) + KL Divergence

**Data Loading**

- Num Workers: 4 (parallel data loading)
- Pin Memory: Enabled (GPU optimization)
- Drop Last Batch: Enabled (consistent batch sizes)

**Hardware Requirements**

- Minimum GPU Memory: 8 GB VRAM (for batch size 128)
- Recommended GPU: NVIDIA RTX 2060 or better
- Training Time: ~15-20 hours on single GPU

## Future Enhancements

- Support for additional facial attributes (from 40 available)
- Real-time attribute editing interface
- Improved image quality with higher resolution models (128×128, 256×256)
- Interactive web application for face editing
- StyleGAN2 integration for enhanced image quality
- Disentangled representation learning


## References

- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114
- Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep Learning Face Attributes in the Wild. ICCV
- Yan, X., Yang, J., Sohn, K., & Lee, H. (2016). Attribute2Image: Conditional Image Generation from Visual Attributes. ECCV
- Sohn, K., Lee, H., & Yan, X. (2015). Learning Structured Output Representation using Deep Conditional Generative Models. NIPS
