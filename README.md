# Network Depth, Augmentation & Background Removal in Lightweight CNNs for Plant Disease Detection

> *An empirical ablation study on model design choices for efficient, deployable crop disease classifiers — trained on a 4GB consumer GPU, ready for edge inference.*

---

## What This Study Found

Most plant disease detection projects pick one model and report one number. This project asks a different question: **what actually drives accuracy — and what breaks it?**

Three factors were systematically isolated across two datasets representing opposite ends of the data spectrum:

**1. Network depth must match dataset size — or the model collapses.**
A 22-layer CNN trained on 2,152 potato images didn't just underperform. It collapsed to 33.3% accuracy — random guessing across three classes. The 7-layer model hit 96.67% on the same data. Depth isn't better; it's contextual.

**2. Augmentation + attention recover what data scarcity loses.**
Adding squeeze-and-excitation (SE) attention blocks and a targeted augmentation pipeline to the 7-layer potato model pushed accuracy from 96.67% → **97.8% (Macro F1 = 0.96)** — within 7.39M parameters and 53ms inference on a 4GB GPU.

**3. Background removal creates a domain alignment requirement.**
U²-Net background removal on the tomato dataset produced a model that dropped from 85.99% to 78.9% when tested on raw field images (BR→Orig). The same model, when test images were also preprocessed (BR→BR), recovered robustness — demonstrating that the preprocessing protocol is as important a design decision as the architecture.

---

## Ablation Summary

### Depth vs. Dataset Size

| Dataset | Volume | Architecture | Accuracy | Observation |
|---|---|---|---|---|
| Potato | 2,152 images | 7-layer (Shallow) | **96.67%** | Optimal capacity — prevents over-memorization |
| Potato | 2,152 images | 14-layer (Medium) | 94.22% | Slight variance inflation from excess capacity |
| Potato | 2,152 images | 22-layer (Deep) | **33.33%** | **Model collapse** — collapses to random guessing |
| Tomato | 18,160 images | 7-layer (Shallow) | 80.33% | Underfits 10-class morphological diversity |
| Tomato | 18,160 images | 14-layer (Medium) | **85.99%** | Data-capacity balance achieved |

### Optimized Final Models

| Crop | Architecture | Preprocessing | Accuracy | Macro F1 | Parameters | Inference |
|---|---|---|---|---|---|---|
| Potato | 7-layer CNN + SE Attention | Heavy augmentation | **97.80%** | **0.96** | 7.39M | 53.3 ms/img |
| Tomato | 14-layer CNN + SE Attention | U²-Net BR → Orig | **78.90%** | **0.73** | 6.52M | 55.0 ms/img |

### Domain Alignment Effect (Tomato)

| Training Domain | Inference Domain | Accuracy |
|---|---|---|
| Original images | Original images | 85.99% |
| Background-removed (BR) | Original images | 78.90% ← **domain mismatch** |
| Background-removed (BR) | Background-removed (BR) | Superior robustness ← **aligned deployment** |

---

## Pipeline Architecture

The full inference pipeline operates in three sequential stages:

```
Input Image
    │
    ▼
[Stage 1] YOLOv8 Leaf Localization
    │   Crops the leaf bounding box, removes extraneous scene clutter
    │
    ▼
[Stage 2] U²-Net Background Segmentation
    │   Pixel-wise saliency map isolates leaf structure from soil/shadows
    │   (Optional — enables BR→BR deployment alignment)
    │
    ▼
[Stage 3] CNN + SE Attention Classification
        7-layer (Potato) or 14-layer (Tomato) with mid-level Squeeze-and-Excitation blocks
        Grad-CAM confirms attention focuses on lesion regions, not background
```

**Hardware:** All training and experiments conducted on NVIDIA RTX 2050 (4GB VRAM), Intel Core i5-11260H, 8GB RAM — demonstrating feasibility on constrained hardware.

---

## Comparison with Related Work

| Study | Crop | Model | Accuracy | Notes |
|---|---|---|---|---|
| **This work** | Potato | 7-layer CNN + SE Attention | **97.8% / F1 0.96** | SOTA, 7.39M params |
| **This work** | Tomato | 14-layer CNN + SE Attention | 78.9% / F1 0.73 | Competitive under domain alignment |
| BSPP Journal (2025) | Tomato | CNN baseline | 75–85% | Realistic field conditions |
| AgriFusionNet (2025) | Tomato, Grape | CNN + multimodal fusion | 93–95% | SOTA, but computationally heavy |
| Nagpal et al. (2024) | Wheat, Barley | CNN–RNN hybrid | 92% / F1 0.90 | Sequential modeling, heavier |
| Springer Review (2024) | Potato, Tomato | CNN variants | 85–99% | Dataset-dependent |

Our potato model matches or exceeds SOTA while remaining under 7.5M parameters. The tomato model is competitive with field-condition benchmarks, without multimodal sensor inputs.

---

## Datasets

**Potato** — 2,152 images across 3 classes *(Early Blight, Late Blight, Healthy)*
Augmented to 18,605 images for training using spatial rotations (±25°), horizontal/vertical flips, zoom (±20%), brightness/contrast shifts, and Gaussian noise.

**Tomato** — 18,160 images across 10 classes *(Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy)*
Background-removed variant created using U²-Net (~18,000 images) to enable domain alignment experiments.

Both datasets split 70% train / 15% validation / 15% test.

---

## Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 with plateau decay |
| Loss | Categorical cross-entropy |
| Batch size | 32 |
| Max epochs | 25 (early stopping on val accuracy) |
| Image size | 224 × 224 |
| Frameworks | TensorFlow/Keras (CNNs), PyTorch (U²-Net) |

---

## Streamlit Application

A full interactive interface is included for exploring results, uploading leaf images through the pipeline, and visualizing Grad-CAM outputs.

<details>
  <summary>🏠 <b>Home Page & Landing Dashboard</b></summary>
  <br>
  <p align="center">
    <img src="assets/screenshots/home_1.png" width="85%" alt="Home Page 1"><br><br>
    <img src="assets/screenshots/home_2.png" width="85%" alt="Home Page 2"><br><br>
    <img src="assets/screenshots/home_3.png" width="85%" alt="Home Page 3"><br><br>
    <img src="assets/screenshots/detection.png" width="85%" alt="Detection Page">
  </p>
</details>

<details>
  <summary>🥔 <b>Potato Pipeline & Diagnosis</b></summary>
  <br>
  <p align="center">
    <img src="assets/screenshots/potato_1.png" width="85%" alt="Potato Page 1"><br><br>
    <img src="assets/screenshots/potato_2.png" width="85%" alt="Potato Page 2"><br><br>
    <img src="assets/screenshots/potato_3.png" width="85%" alt="Potato Page 3">
  </p>
</details>

<details>
  <summary>🍅 <b>Tomato Pipeline & Diagnosis</b></summary>
  <br>
  <p align="center">
    <img src="assets/screenshots/tomato_1.png" width="85%" alt="Tomato Page 1"><br><br>
    <img src="assets/screenshots/tomato_2.png" width="85%" alt="Tomato Page 2"><br><br>
    <img src="assets/screenshots/tomato_3.png" width="85%" alt="Tomato Page 3">
  </p>
</details>

<details>
  <summary>📊 <b>Model Analysis & Grad-CAM Dashboards</b></summary>
  <br>
  <p align="center">
    <img src="assets/screenshots/gradcam.png" width="85%" alt="Grad-CAM Outputs">
  </p>
</details>

---

## Setup

### 1. Clone
```bash
git clone https://github.com/Tarun995/Lightweight-CNN-Attention-for-Plant-Disease-Detection.git
cd Lightweight-CNN-Attention-for-Plant-Disease-Detection
```

### 2. Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 3. Download Model Weights

Weight files (`.keras`, `.pth`, `.pt`) are hosted on Google Drive due to size.

**Automated download:**
```powershell
# PowerShell
.\scripts\download_weights_gdown.ps1
```
```bash
# Bash / macOS / Linux
bash scripts/download_weights_gdown.sh
```

**Manual:** [Google Drive folder](https://drive.google.com/drive/folders/11f_gsqnwHH1Iq9WUzKZNqF8IJ3pRWoeR?usp=drive_link) → extract into `models/`

```
models/
├── disease/   potato.keras  tomato.keras  maize.keras
├── yolo/      yolo_leaf_best.pt  yolo_lesion_best.pt
└── u2net/     u2net.pth
```

### 4. Run
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501)

---

## Project Structure

```
├── app.py                        ← Streamlit entrypoint
├── pages/
│   ├── 2_Detection.py            ← Image upload & pipeline diagnosis
│   ├── 3_Model_Analysis.py       ← Confusion matrices, evaluation reports
│   ├── 4_Research_Dashboard.py   ← Baseline vs. attention comparison
│   ├── 5_GradCAM.py              ← Grad-CAM interpretability heatmaps
│   └── 6_About.py                ← References and contributors
├── utils/
│   ├── attention_layer.py        ← SE block implementation
│   ├── disease_info.py           ← Class labels and treatment guides
│   ├── model_loader.py           ← Cached model loading
│   ├── pipeline.py               ← Full inference pipeline
│   └── u2net_bg_removal.py       ← U²-Net preprocessing helper
├── models/                       ← Weights directory (gitignored)
├── results/
│   ├── confusion_matrices/       ← Per-model validation matrices
│   ├── evaluation/               ← Classification report CSVs
│   └── gradcam/                  ← Pre-computed Grad-CAM outputs
└── requirements.txt
```

---

## Key Takeaways for Practitioners

- **Don't default to deep models.** On datasets under ~5K images, a 7-layer CNN with augmentation will outperform a 22-layer model by 60+ percentage points.
- **Preprocessing protocol is a hyperparameter.** If you train on background-removed images, you must deploy on background-removed images — or accept the accuracy penalty.
- **SE attention adds almost nothing to parameter count.** Going from 7-layer baseline to 7-layer + SE attention costs negligible parameters but recovers measurable F1.
- **Edge deployment is viable.** Sub-60ms inference on a 4GB GPU means this pipeline runs on hardware accessible to agricultural AI applications.

---

*Associated research paper: "The Role of Network Depth, Data Augmentation, and Background Removal in Lightweight CNNs for Plant Disease Detection"*
