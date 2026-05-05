# 🌑 CTU-Net+ Shadow Detection & Removal

A deep learning web application that automatically detects and removes shadows from images using the **CTU-Net+** architecture — a CNN-Transformer hybrid with pixel-wise learned illumination compensation.

> **Live Demo:** [Link once deployed]

---

## 📸 How It Works

Upload any image → the model detects shadow regions → removes them → shows you the result.

| Input (Shadow) | Shadow Mask | Output (Shadow-Free) |
|:-:|:-:|:-:|
| ![](assets/sample_input.jpg) | ![](assets/sample_mask.png) | ![](assets/sample_output.png) |

---

## 🧠 Architecture

Based on the paper:
> *Shadow Detection and Removal on Images: CTUNet+ with Pixel-wise Learned Illumination Compensation and Perceptual Boundary Refinement*

The pipeline has four main modules:

| Module | Role |
|---|---|
| **CTU-Net+** | CNN-Transformer hybrid shadow detector with material-aware branch |
| **LICM** | Learned Illumination Compensation Module — pixel-wise (not region-wise) |
| **ARSM** | Adaptive Region Segmentation & Learned Matching for lighting transfer |
| **PBR** | Perceptual Boundary Refinement using a conditional GAN |

---

## 🗂️ Project Structure

my_shadow_project/
├── checkpoints/           # Model weights (auto-downloaded on first run)
├── shadow_model/
│   ├── init.py
│   ├── architecture.py    # All model classes (CTUNetPlus, LICM, ARSM, PBR...)
│   └── inference.py       # load_model() and remove_shadow() functions
├── main.py                # Flask web app
├── requirements.txt
└── README.md

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/LakshyaPRJ/shadow-removal-ctunet
cd shadow-removal-ctunet

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app (model downloads automatically on first run)
python main.py

# 4. Open in browser
# → http://127.0.0.1:5000
```

---

## 📊 Results

Trained on a subset (300 images) of the [ISTD dataset](https://github.com/DeepInsight-PCALab/ST-CGAN) for 30 epochs on Google Colab (free tier, T4 GPU).

| Metric | Our Result | Paper (full training) |
|---|---|---|
| BER ↓ | 2.351 | 1.43 |
| RMSE ↓ | 0.0797 | — |

### ⚠️ Results Can Be Improved

The current model was trained on a **small subset** of the ISTD dataset (300 of 1330 images) for only **30 epochs** due to Google Colab free tier limitations. Results will improve significantly with:

- **Full ISTD dataset** (all 1330 training images)
- **More epochs** (paper uses 200)
- **Larger image resolution** (paper uses 640×480 vs our 256×256)
- **Richer datasets** such as [SRD](https://github.com/Liangqiong/DeShadowNet), [AISTD](https://paperswithcode.com/dataset/aistd), or synthetic shadow datasets
- Training on a **GPU with more VRAM** (e.g. A100, V100)

The architecture itself is capable of strong results — the bottleneck is training data and compute, not the model design.

---

## 🛠️ Tech Stack

- **PyTorch** — model training & inference
- **Flask** — web server
- **einops** — tensor operations
- **Pillow** — image handling
- **Google Colab** (T4 GPU) — training environment
- **Dataset** — ISTD (Image Shadow Triplets Dataset)

---

## 👤 Author

**Lakshya Prajapati**
- GitHub: [@LakshyaPRJ](https://github.com/LakshyaPRJ)

---

## 📄 License

MIT License — free to use, modify, and distribute.