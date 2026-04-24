# 🔬 SkinScan AI — HAM10000 Skin Cancer Classifier

A deep learning web application for dermoscopy image classification using a **ConvNeXt-Small** backbone trained on the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection). It classifies skin lesions into 7 categories with **10-pass Test-Time Augmentation (TTA)** and **Grad-CAM++** visual explanations.

> ⚠️ **Disclaimer:** For research and educational purposes only. Not a substitute for professional clinical diagnosis.

---

## ⚠️ Important — Model Checkpoint Not Included

> **The trained model file (`convnext_best.pth`) is NOT included in this repository.**
>
> GitHub has a 100 MB file size limit, and the trained checkpoint exceeds that. You must **train the model yourself on Google Colab** before you can run this app. Full step-by-step instructions are in the [Training on Google Colab](#-training-on-google-colab) section below.

---

## ✨ Features

- **7-class classification** of dermoscopic images
- **ConvNeXt-Small** backbone with a custom multi-layer classification head
- **DullRazor hair removal** preprocessing pipeline
- **10-pass Test-Time Augmentation** for robust predictions
- **Grad-CAM++** attention heatmaps to highlight regions influencing the prediction
- Clean, dark-themed **web UI** served directly from the FastAPI backend
- REST API with interactive **Swagger docs** at `/docs`

---

## 🗺️ How This Project Works — Overview

```
Step 1: Train model on Google Colab   →   convnext_best.pth  (downloaded to your machine)
Step 2: Place checkpoint in repo      →   backend/models/convnext_best.pth
Step 3: Run the app                   →   http://localhost:8000
```

You only need to train once. After that, the checkpoint stays on your machine and the app works offline.

---

## 🏷️ Supported Classes

| Code | Condition | Risk Level |
|------|-----------|------------|
| `akiec` | Actinic Keratoses | Pre-cancerous |
| `bcc` | Basal Cell Carcinoma | Malignant |
| `bkl` | Benign Keratosis | Benign |
| `df` | Dermatofibroma | Benign |
| `mel` | Melanoma | Malignant |
| `nv` | Melanocytic Nevi | Benign |
| `vasc` | Vascular Lesions | Benign |

---

## 📁 Project Structure

```
skinscan-ai/
├── backend/
│   ├── main.py              # FastAPI app & routes
│   ├── model.py             # ConvNeXt model, Grad-CAM++, inference engine
│   ├── requirements.txt     # Python dependencies
│   └── models/
│       └── convnext_best.pth   # ← YOU must place your trained checkpoint here
│                                #   (not in repo — train it first, see below)
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── run.sh                   # One-click launcher (Linux / macOS)
└── run.bat                  # One-click launcher (Windows)
```

---

## 🧑‍💻 Training on Google Colab

Training takes roughly **1–2 hours on a free T4 GPU**. Follow each step in order.

---

### Step 1 — Open Google Colab and enable GPU

Go to [https://colab.research.google.com](https://colab.research.google.com) and sign in.

Then: **Runtime → Change runtime type → T4 GPU → Save**

---

### Step 2 — Download the HAM10000 dataset

The dataset is on Kaggle. You'll need a free Kaggle account and API key.

1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**. This downloads `kaggle.json`.
2. Run these cells in Colab:

```python
# Install Kaggle API
!pip install kaggle -q

# Upload your kaggle.json
from google.colab import files
files.upload()   # select the kaggle.json you downloaded

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download and unzip HAM10000
!kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection
!unzip -q skin-lesion-analysis-toward-melanoma-detection.zip -d ham10000/
!ls ham10000/
```

> **Alternative:** Download the zip manually from Kaggle, upload it to Google Drive, then mount Drive in Colab and unzip from there.

---

### Step 3 — Install dependencies

```python
!pip install timm -q
# torch, torchvision, numpy, pandas are pre-installed in Colab
```

---

### Step 4 — Define the model

> ⚠️ **This architecture must exactly match `model.py` in this repo.** Do not change layer sizes or names, or you will get errors when loading the checkpoint into the app.

```python
import torch
import torch.nn as nn
import timm

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes=7, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model(
            'convnext_small',
            pretrained=True,        # ImageNet weights as starting point
            num_classes=0,
            global_pool='avg',
        )
        feat_dim = self.backbone.num_features   # 768 for convnext_small

        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout / 4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))

model = ConvNeXtClassifier(num_classes=7, dropout=0.4).cuda()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

### Step 5 — Set up data loading

```python
import os, pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

IMG_SIZE = 300

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class HAM10000Dataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform
        self.label_map = {c: i for i, c in enumerate(CLASSES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = os.path.join(self.img_dir, row['image_id'] + '.jpg')
        img   = Image.open(path).convert('RGB')
        label = self.label_map[row['dx']]
        if self.transform:
            img = self.transform(img)
        return img, label

meta     = pd.read_csv('ham10000/HAM10000_metadata.csv')
train_df, val_df = train_test_split(meta, test_size=0.2, stratify=meta['dx'], random_state=42)

# Adjust this path if your images extracted to a different folder
IMG_DIR = 'ham10000/ham10000_images_part1'

train_ds     = HAM10000Dataset(train_df, IMG_DIR, train_transform)
val_ds       = HAM10000Dataset(val_df,   IMG_DIR, val_transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
```

---

### Step 6 — Train the model

```python
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce   = F.cross_entropy(inputs, targets, reduction='none')
        pt   = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
criterion = FocalLoss(gamma=2.0).cuda()

EPOCHS   = 20
best_acc = 0.0

for epoch in range(EPOCHS):
    # Train
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.cuda(), labels.cuda()
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    # Validate
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            out = model(imgs)
            val_correct += (out.argmax(1) == labels).sum().item()
            val_total   += labels.size(0)
    val_acc = val_correct / val_total

    scheduler.step()
    print(f"Epoch {epoch+1:02d}/{EPOCHS}  "
          f"Loss: {total_loss/len(train_loader):.4f}  "
          f"Train: {correct/total:.4f}  Val: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'convnext_best.pth')
        print(f"  ✅ New best saved (val acc: {val_acc:.4f})")

print(f"\nDone. Best val accuracy: {best_acc:.4f}")
```

---

### Step 7 — Download the checkpoint

**Option A — Direct download to your computer:**
```python
from google.colab import files
files.download('convnext_best.pth')
```

**Option B — Save to Google Drive (safer if Colab disconnects):**
```python
from google.colab import drive
drive.mount('/content/drive')
import shutil
shutil.copy('convnext_best.pth', '/content/drive/MyDrive/convnext_best.pth')
print("Saved to Google Drive.")
```

---

### Step 8 — Place the checkpoint in the repo

Move the downloaded file into the `backend/models/` folder:

```bash
# Create the folder if it doesn't exist
mkdir -p backend/models

# Move the checkpoint (adjust source path as needed)
mv ~/Downloads/convnext_best.pth backend/models/convnext_best.pth
```

Verify it's in the right place:
```bash
ls -lh backend/models/convnext_best.pth
```

You're now ready to run the app. ✅

---

## 🚀 Running the App

### Option 1 — One-click launch (recommended)

**Linux / macOS**
```bash
chmod +x run.sh
./run.sh
```

**Windows**
```
Double-click run.bat
# or run from cmd / PowerShell
```

Both scripts automatically create a Python virtual environment, install all dependencies, and start the server.

---

### Option 2 — Manual setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/skinscan-ai.git
cd skinscan-ai/backend

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Confirm checkpoint exists
ls models/convnext_best.pth

# 5. Launch
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🌐 Accessing the App

| URL | Description |
|-----|-------------|
| `http://localhost:8000` | Web UI |
| `http://localhost:8000/docs` | Interactive API docs (Swagger) |
| `http://localhost:8000/health` | Health check & model status |
| `http://localhost:8000/classes` | Class metadata JSON |

---

## 🔌 API Reference

### `POST /predict`

| Field | Type | Description |
|-------|------|-------------|
| `file` | `File` | JPG or PNG image |
| `tta_passes` | `int` | TTA passes (1–10, default: 10) |

**Response JSON:**
```json
{
  "predictions": [{ "class": "mel", "name": "Melanoma", "probability": 0.87, "risk": "Malignant" }],
  "top_class": "mel",
  "top_prob": 0.87,
  "risk": "Malignant",
  "gradcam_overlay_b64": "<base64 JPEG>",
  "hair_removed_b64": "<base64 JPEG>",
  "original_b64": "<base64 JPEG>"
}
```

### `GET /health` — Model load status
### `GET /classes` — Metadata for all 7 classes

---

## 🏗️ Model Architecture

```
ConvNeXt-Small backbone (timm, global_pool='avg') → 768-dim features
LayerNorm → Dropout(0.4) → Linear(768→1024) → GELU
          → Dropout(0.2) → Linear(1024→512)  → GELU
          → Dropout(0.1) → Linear(512→7)
```

---

## 🛠️ Troubleshooting

**`/predict` returns 503**
→ Checkpoint is missing. Confirm `backend/models/convnext_best.pth` exists and restart.

**Architecture mismatch errors on startup**
→ The Colab model definition must exactly match `model.py`. Re-copy the architecture from Step 4 of the training guide.

**Colab disconnected before training finished**
→ Mount Google Drive and save checkpoints there to avoid losing progress on disconnect.

**Slow inference (~5s)**
→ Normal for 10-pass TTA on CPU. Lower the TTA slider in the UI, or use a CUDA GPU.

**Port already in use**
→ Run `uvicorn main:app --port 8001` or edit the port in `run.sh` / `run.bat`.

**`cv2` import error**
→ Use `opencv-python-headless`, not `opencv-python`.

---

## 📄 License

This project is for educational and research use. The HAM10000 dataset is subject to its own [license terms](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection).
