#  Smart Plant Disease Detection System

A multi-stage deep learning pipeline combining YOLOv8 object detection, U²-Net background segmentation, and CNNs with custom Attention layers to diagnose plant leaf diseases with high precision.

---

##  Key Features
- **Leaf Localization**: YOLOv8 extracts leaf bounding boxes to crop extraneous background noise.
- **Salient Segmentation**: U²-Net computes pixel-wise saliency maps to isolate the leaf structure from soil, neighboring plants, or shadows.
- **Lesion Isolation**: YOLOv8 flags specific lesion hotspots to create a high-contrast binary mask input.
- **Attention-Augmented Classification**: Custom self-attention layers dynamically prioritize disease-bearing features during classification.
- **Model Explainability**: Integrated **Grad-CAM** visualizations highlight activation areas showing model decision-making.

---

##  Project Structure
```
d:\Projects\PlantDiseaseDetection\
├── app.py                          ← Home/Landing page (Streamlit entrypoint)
├── pages/
│   ├── 2_Detection.py              ← Interactive image upload and pipeline diagnosis
│   ├── 3_Model_Analysis.py         ← Evaluation reports, confusion matrices
│   ├── 4_Research_Dashboard.py     ← Baseline CNN vs. Attention comparison
│   ├── 5_GradCAM.py                ← Grad-CAM interpretability heatmaps
│   └── 6_About.py                  ← Contributors, references, links
├── utils/
│   ├── __init__.py
│   ├── attention_layer.py          ← Canonical custom AttentionLayer serialization
│   ├── disease_info.py             ← Actionable guides, labels, links
│   ├── model_loader.py             ← Cached model loader (on-demand loading)
│   ├── pipeline.py                 ← Vision pipeline execution
│   └── u2net_bg_removal.py         ← U²-Net background extraction helper
├── models/
│   ├── disease/                    ← Potato, Tomato, Corn Keras models (gitignored)
│   ├── yolo/                       ← YOLO Leaf and Lesion weight files (gitignored)
│   └── u2net/                      ← U²-Net structure definition and weights (gitignored)
├── results/
│   ├── confusion_matrices/         ← Validation confusion matrix images
│   ├── evaluation/                 ← Model classification report CSV files
│   └── gradcam/                    ← Pre-computed sample Grad-CAM outputs
├── requirements.txt                ← Project dependencies
├── README.md                       ← This documentation file
└── .gitignore                      ← Ignores cache, virtual environments, and large models
```

---

##  Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/username/PlantDiseaseDetection.git
cd PlantDiseaseDetection
```

### 2. Configure Virtual Environment
We recommend using a clean virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Weights Setup

Model weight files (`.keras`, `.pth`, `.pt`) are large and hosted externally on Google Drive. Viewer access is enabled for public download.

**Google Drive Folder:**
https://drive.google.com/drive/folders/11f_gsqnwHH1Iq9WUzKZNqF8IJ3pRWoeR?usp=drive_link

**Quick Setup (Automated Download)**

1. Install `gdown` (if not already installed):
```bash
pip install gdown
```

2. Run the download script from the repository root:

**PowerShell:**
```powershell
.\scripts\download_weights_gdown.ps1
```

**Bash / macOS / Linux:**
```bash
bash scripts/download_weights_gdown.sh
```

The script will download all weights from Google Drive, organize them into `models/disease/`, `models/yolo/`, and `models/u2net/`, compute checksums, and optionally commit locally.

**Manual Download (via Google Drive UI)**

Alternatively, visit the Google Drive folder above, download the ZIP, and extract into the `models/` directory:
```
models/
├── disease/
│   ├── potato.keras
│   ├── tomato.keras
│   └── maize.keras
├── yolo/
│   ├── yolo_leaf_best.pt
│   └── yolo_lesion_best.pt
└── u2net/
    └── u2net.pth
```

See [WEIGHTS_URLS.md](WEIGHTS_URLS.md) for more details.

---

##  Running the Application

Launch the Streamlit web interface locally:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

##  Core Finding: Depth vs. Dataset Size (Ablation Study)

A major focus of this project is investigating the structural trade-off between convolutional neural network (CNN) depth and dataset scale. We evaluated 7-layer (shallow), 14-layer (medium), and 22-layer (deep) architectures trained from scratch across two highly distinct data regimes to observe how network capacity interacts with sample volume:

| Dataset (Volume) | Model Architecture | Accuracy (%) | Empirical Observation & Phenomenon |
|---|---|---|---|
| **Potato** (2,152 images) | 7-layer (Shallow) | **96.67%** | **Optimal Capacity:** Strong generalizability; restricted parameters prevent feature over-memorization on limited samples. |
| **Potato** (2,152 images) | 14-layer (Medium) | 94.22% | **Variance Inflation:** Increased capacity introduces slight overfitting to non-essential background artifacts. |
| **Potato** (2,152 images) | 22-layer (Deep) | 33.33% | **Model Collapse:** Severe overfitting and gradient issues cause the network to fail entirely, collapsing to random guessing (1/3 classes). |
| **Tomato** (18,160 images) | 7-layer (Shallow) | 80.33% | **Underfitting:** Constrained representational capacity fails to map fine-grained lesion features across 10 complex classes. |
| **Tomato** (18,160 images) | 14-layer (Medium) | **85.99%** | **Data-Capacity Balance:** Higher structural depth successfully models the vast morphological diversity of the larger dataset. |

**Conclusion:** Deeper networks are not universally superior. While an expanded feature extraction stack benefits high-volume, multi-class tasks (Tomato), it can induce total model collapse on localized data regimes (Potato). In data-constrained agricultural applications, a carefully regulated shallow architecture provides superior validation safety.

---

##  Final Optimized Model Performance

To push past baseline performance limits, we integrated customized data workflows and attention mechanisms. For the Potato dataset, we introduced a heavy conditional data augmentation pipeline (spatial rotations, contrast adjustments, and Gaussian noise) to scale training volume. Both architectures were then optimized with mid-level **Squeeze-and-Excitation (SE) blocks** to dynamically recalibrate feature maps.

The final benchmark results of our optimized multi-stage pipeline are structured below:

| Target Crop | Core Network Architecture | Preprocessing & Augmentation | Accuracy | Macro F1 | Total Parameters | Inference Latency |
|---|---|---|---|---|---|---|
| **Potato** | 7-Layer CNN + SE Attention | Heavy Augmentation | **97.80%** | **0.96** | 7.39M | 53.3 ms / image |
| **Tomato** | 14-Layer CNN + SE Attention | U²-Net Background Removed (BR → Orig) | **78.90%** | **0.73** | 6.52M | 55.0 ms / image |

####  Critical Insight: Preprocessing Domain Shift
* **The Domain Mismatch Challenge (BR → Orig):** Training our 14-layer Tomato model on background-removed (BR) leaf images isolates clean structural boundaries but trains model weights strictly on foreground pixels. When evaluating this specific model configuration against raw test data containing unsegmented field clutter (soil, weeds, or raw illumination variables), accuracy drops from 85.99% to 78.90%.
* **The Alignment Solution (BR → BR):** Despite the numerical drop when introduced to raw test images, background-removed training establishes significantly higher functional robustness when deployment testing domains are perfectly aligned. By passing production field data through the full multi-stage pipeline—where input images are sequentially cropped via YOLO for leaf tracking and segmented via U²-Net for background extraction before classification—we neutralize background dependency and eliminate misleading field noise correlations.
