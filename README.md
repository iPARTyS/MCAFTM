# MCAFT: Multi-scale Convolutional Attention Frequency-enhanced Transformer Network for Medical Image Segmentation

*Figure 1: Overview of the MCAFT architecture.*  
![R_model](https://github.com/user-attachments/assets/efc3e44d-bc93-4990-8f72-7eb466bc2909)

## 📌 Overview
[MCAFT](https://www.sciencedirect.com/science/article/abs/pii/S1566253525000922?via%3Dihub) is an advanced transformer-based network designed for medical image segmentation, addressing limitations in existing methods such as insufficient local feature extraction and loss of high-frequency details. By integrating **wavelet transform** with **multi-scale convolutional attention**, MCAFT achieves superior performance in preserving edge and texture information while capturing global context.

---

## 🚀 Key Innovations
- **Multi-Scale Convolutional Attention Frequency-enhanced Transformer Module (MCAFTM)**:
  - Combines channel and spatial attention mechanisms
  - Uses Discrete Wavelet Transform (DWT) to decompose features into low-frequency (LL) and high-frequency (LH, HL, HH) sub-bands
  - Enhances boundary information through reverse attention

- **Efficient Frequency-enhanced Transformer Module (EFTM)**:
  - Applies wavelet transform for reversible downsampling
  - Preserves high-frequency details through multi-resolution analysis
  - Reduces computational complexity with efficient attention mechanisms

- **Multi-Scale Progressive Gate-Spatial Attention (MSGA)**:
  - Facilitates information flow between encoder and decoder
  - Uses gating signals to regulate feature integration
  - Enhances local contextual relationships with large kernel convolutions

---

## 📊 Performance Highlights
### Synapse Multi-Organ CT Dataset
| Metric       | MCAFT | TransUNet | SwinUNet | PVT-EMCAD |
|--------------|-------|-----------|----------|-----------|
| **Dice (%)** | **83.87** | 77.61 | 77.58 | 83.63 |
| **HD95**     | **14.20** | 26.90 | 27.32 | 15.68 |
| **mIoU (%)** | **74.73** | 75.00 | 75.79 | 83.92 |

### ACDC Cardiac MRI Dataset
| Metric       | MCAFT | TransUNet | SwinUNet | PVT-EMCAD |
|--------------|-------|-----------|----------|-----------|
| **Dice (%)** | **92.32** | 89.71 | 88.07 | 92.12 |
| **RV Dice**  | **91.07** | 86.67 | 85.77 | 90.65 |
| **LV Dice**  | **96.05** | 95.18 | 94.03 | 96.02 |

### Polyp Segmentation Datasets
| Dataset      | ClinicDB | Kvasir | ColonDB | ETIS | CVC-T |
|--------------|----------|--------|---------|------|-------|
| **Dice (%)** | **94.49** | **92.62** | 81.07 | **78.68** | 88.91 |

---

## 🛠️ Implementation Details
1. **Architecture**:
   - Encoder: 4-level PVT-V2 backbone
   - Decoder: MCAFTD with MCAFTM and MSGA modules
   - Segmentation heads for multi-scale output

2. **Training**:
   - Optimizer: AdamW (lr=0.0001)
   - Batch size: 8 (Synapse), 12 (ACDC)
   - Input resolution: 224×224 (Synapse/ACDC), 352×352 (Polyp)
   - Data augmentation: Random rotation and flipping

3. **Wavelet Configuration**:
   - Coiflet wavelet function for optimal detail preservation
   - 4 sub-bands: LL (low-frequency), LH/HL/HH (high-frequency)

---

## 🏆 Comparative Results
### Visual Comparisons
*Figure 2: Qualitative results on Synapse dataset showing superior boundary precision.*
![r_s](https://github.com/user-attachments/assets/51289598-9af7-422d-a37e-c44628c848f8)

*Figure 3: Precise segmentation of small polyps and complex boundaries.*
![r_polyp](https://github.com/user-attachments/assets/235ac4ef-5bfd-4aa7-a104-1e3ff51fe2cf)

### Ablation Studies
| Configuration | Dice (%) | mIoU (%) | HD95 |
|---------------|----------|----------|------|
| Full MCAFT    | **83.87** | **74.73** | **14.20** |
| w/o EFTM      | 82.95 (-0.92) | 73.58 (-1.15) | 14.51 |
| w/o MSGA      | 83.08 (-0.79) | 73.82 (-0.91) | 15.20 |
| w/o MCAFTM    | 82.34 (-1.53) | 74.31 (-0.42) | 16.26 |

---

## 📜 Citation
```bibtex
@article{yan2025mcaft,
  title={Multi-scale Convolutional Attention Frequency-enhanced Transformer Network for Medical Image Segmentation},
  author={Yan, Shun and Yang, Benquan and Chen, Aihua and Zhao, Xiaoming and Zhang, Shiqing},
  journal={Information Fusion},
  volume={119},
  pages={103019},
  year={2025},
  publisher={Elsevier}
}
