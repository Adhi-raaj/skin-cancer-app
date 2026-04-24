"""
model.py — ConvNeXt-Small classifier + Grad-CAM++ inference
Must match the architecture used in the training notebook exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import cv2
from pathlib import Path

# ── Class definitions ────────────────────────────────────────────────────────

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

CLASS_INFO = {
    'akiec': {
        'name':  'Actinic Keratoses',
        'risk':  'Pre-cancerous',
        'color': '#F39C12',
        'desc':  'Rough, scaly patches caused by sun damage. Can progress to squamous cell carcinoma.',
    },
    'bcc': {
        'name':  'Basal Cell Carcinoma',
        'risk':  'Malignant',
        'color': '#E74C3C',
        'desc':  'Most common skin cancer. Rarely spreads but can cause local damage if untreated.',
    },
    'bkl': {
        'name':  'Benign Keratosis',
        'risk':  'Benign',
        'color': '#2ECC71',
        'desc':  'Non-cancerous skin growth including seborrheic keratoses and solar lentigines.',
    },
    'df': {
        'name':  'Dermatofibroma',
        'risk':  'Benign',
        'color': '#1ABC9C',
        'desc':  'Harmless fibrous nodule, usually appearing on the legs.',
    },
    'mel': {
        'name':  'Melanoma',
        'risk':  'Malignant',
        'color': '#C0392B',
        'desc':  'Most dangerous skin cancer. Early detection is critical for survival.',
    },
    'nv': {
        'name':  'Melanocytic Nevi',
        'risk':  'Benign',
        'color': '#3498DB',
        'desc':  'Common moles formed by clusters of pigmented cells. Usually harmless.',
    },
    'vasc': {
        'name':  'Vascular Lesions',
        'risk':  'Benign',
        'color': '#9B59B6',
        'desc':  'Growths involving blood vessels such as angiomas and pyogenic granulomas.',
    },
}


# ── Model Architecture ───────────────────────────────────────────────────────

class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXt-Small classifier — must exactly match the training notebook.
    Backbone: timm convnext_small, global_pool='avg'
    Head: LayerNorm → Dropout(0.4) → Linear(768→1024) → GELU
          → Dropout(0.2) → Linear(1024→512) → GELU
          → Dropout(0.1) → Linear(512→7)
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.4):
        super().__init__()
        self.backbone = timm.create_model(
            'convnext_small',
            pretrained=False,   # weights loaded from checkpoint
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


# ── Grad-CAM++ ───────────────────────────────────────────────────────────────

class GradCAMPlusPlus:
    """Grad-CAM++ on the last ConvNeXt stage."""

    def __init__(self, model: ConvNeXtClassifier):
        self.model       = model
        self.gradients   = None
        self.activations = None

        # Hook last stage of ConvNeXt backbone
        target_layer = list(model.backbone.stages.children())[-1]
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
        device: torch.device = torch.device('cpu'),
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """
        Returns:
            cam      — (H, W) float32 array, values in [0, 1]
            class_idx — predicted class index
            probs     — (7,) softmax probabilities
        """
        self.model.eval()
        inp = input_tensor.unsqueeze(0).to(device)
        inp.requires_grad_(True)

        out = self.model(inp)
        if class_idx is None:
            class_idx = out.argmax(1).item()

        self.model.zero_grad()
        out[0, class_idx].backward()

        grads = self.gradients    # (1, C, H, W)
        acts  = self.activations  # (1, C, H, W)

        # Grad-CAM++ weights
        alpha   = grads.pow(2) / (
            2 * grads.pow(2)
            + acts * grads.pow(3).sum(dim=[2, 3], keepdim=True)
            + 1e-8
        )
        weights = (alpha * torch.relu(grads)).sum(dim=[2, 3], keepdim=True)
        cam     = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam     = (cam - cam.min()) / (cam.max() + 1e-8)

        probs = out.softmax(1).squeeze().detach().cpu().numpy()
        return cam.squeeze().cpu().numpy(), int(class_idx), probs


# ── Inference Engine ─────────────────────────────────────────────────────────

class SkinCancerModel:
    """Load checkpoint and run full inference with TTA + Grad-CAM++."""

    IMG_SIZE = 300
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = ConvNeXtClassifier(num_classes=7, dropout=0.4)

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {ckpt_path}\n"
                "Train the model first using the Colab notebook, "
                "then place 'convnext_best.pth' in the backend/models/ folder."
            )

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        # Support both raw state dict and wrapped checkpoint
        state = ckpt.get('model_state_dict', ckpt)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.gradcam = GradCAMPlusPlus(self.model)
        print(f"✅ Model loaded on {self.device}")

    # ── Preprocessing helpers ────────────────────────────────────────────

    @staticmethod
    def _remove_hair(img: np.ndarray) -> np.ndarray:
        """DullRazor hair-removal algorithm."""
        gray     = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask  = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        cleaned  = cv2.inpaint(img, mask, inpaintRadius=3,
                               flags=cv2.INPAINT_TELEA)
        return cleaned

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """ImageNet normalization."""
        img = img.astype(np.float32) / 255.0
        mean = np.array(self.MEAN, dtype=np.float32)
        std  = np.array(self.STD,  dtype=np.float32)
        return (img - mean) / std

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        return torch.tensor(
            img.transpose(2, 0, 1), dtype=torch.float32)

    def _augmented_tensor(self, img_clean: np.ndarray, aug_id: int) -> torch.Tensor:
        """Apply one of 10 TTA augmentations."""
        img = img_clean.copy()
        if aug_id == 1:
            img = cv2.flip(img, 1)            # horizontal flip
        elif aug_id == 2:
            img = cv2.flip(img, 0)            # vertical flip
        elif aug_id == 3:
            img = cv2.flip(cv2.flip(img, 1), 0)   # both flips
        elif aug_id == 4:
            img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)   # brighter
        elif aug_id == 5:
            M   = cv2.getRotationMatrix2D(
                (self.IMG_SIZE//2, self.IMG_SIZE//2), 10, 1.0)
            img = cv2.warpAffine(img, M, (self.IMG_SIZE, self.IMG_SIZE))
        elif aug_id == 6:
            M   = cv2.getRotationMatrix2D(
                (self.IMG_SIZE//2, self.IMG_SIZE//2), -10, 1.0)
            img = cv2.warpAffine(img, M, (self.IMG_SIZE, self.IMG_SIZE))
        elif aug_id == 7:
            # CLAHE
            lab  = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            cl   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = cl.apply(lab[:, :, 0])
            img  = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        elif aug_id == 8:
            img = cv2.convertScaleAbs(img, alpha=0.9, beta=-10)  # darker
        elif aug_id == 9:
            img = cv2.flip(img, 1)
            img = cv2.convertScaleAbs(img, alpha=1.05, beta=5)

        norm = self._normalize(img)
        return self._to_tensor(norm)

    # ── Main inference ───────────────────────────────────────────────────

    def predict(
        self,
        image_bytes: bytes,
        tta_passes: int = 10,
    ) -> dict:
        """
        Full inference pipeline.

        Args:
            image_bytes: Raw bytes of the uploaded image.
            tta_passes:  Number of TTA augmentation passes (1–10).

        Returns:
            {
                predictions: [{class, name, probability, risk, color, desc}],
                top_class: str,
                top_name: str,
                top_prob: float,
                risk: str,
                gradcam_overlay_b64: str,   # base64 JPEG
                hair_removed_b64: str,      # base64 JPEG
                original_b64: str,          # base64 JPEG
            }
        """
        # Decode image
        nparr    = np.frombuffer(image_bytes, np.uint8)
        img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image. Upload a valid JPG or PNG.")

        img_orig  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_clean = self._remove_hair(img_orig)
        img_rs    = cv2.resize(img_clean, (self.IMG_SIZE, self.IMG_SIZE))
        img_orig_rs = cv2.resize(img_orig, (self.IMG_SIZE, self.IMG_SIZE))

        # ── TTA inference ────────────────────────────────────────────────
        all_probs = []
        with torch.no_grad():
            for i in range(tta_passes):
                tensor = self._augmented_tensor(img_rs, i).unsqueeze(0).to(self.device)
                with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                    out = self.model(tensor)
                probs = F.softmax(out, dim=1).squeeze().cpu().numpy()
                all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0)
        top_idx   = int(np.argmax(avg_probs))
        top_cls   = CLASSES[top_idx]

        # ── Grad-CAM++ on base (no-aug) tensor ──────────────────────────
        base_tensor = self._to_tensor(self._normalize(img_rs))
        cam, _, _   = self.gradcam.generate(base_tensor, top_idx, self.device)

        # Build overlay
        cam_rs    = cv2.resize(cam, (self.IMG_SIZE, self.IMG_SIZE))
        heat_uint = (cam_rs * 255).astype(np.uint8)
        heat_col  = cv2.applyColorMap(heat_uint, cv2.COLORMAP_JET)
        heat_rgb  = cv2.cvtColor(heat_col, cv2.COLOR_BGR2RGB)
        overlay   = cv2.addWeighted(img_rs, 0.55, heat_rgb, 0.45, 0)

        # ── Encode images to base64 ──────────────────────────────────────
        def to_b64(arr: np.ndarray) -> str:
            import base64
            _, buf = cv2.imencode('.jpg', cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, 90])
            return base64.b64encode(buf).decode('utf-8')

        # ── Build response ───────────────────────────────────────────────
        predictions = []
        for i, (cls, prob) in enumerate(zip(CLASSES, avg_probs)):
            info = CLASS_INFO[cls]
            predictions.append({
                'class':       cls,
                'name':        info['name'],
                'probability': float(prob),
                'risk':        info['risk'],
                'color':       info['color'],
                'desc':        info['desc'],
            })
        # Sort by probability descending
        predictions.sort(key=lambda x: x['probability'], reverse=True)

        top_info = CLASS_INFO[top_cls]
        return {
            'predictions':         predictions,
            'top_class':           top_cls,
            'top_name':            top_info['name'],
            'top_prob':            float(avg_probs[top_idx]),
            'risk':                top_info['risk'],
            'gradcam_overlay_b64': to_b64(overlay),
            'hair_removed_b64':    to_b64(img_rs),
            'original_b64':        to_b64(img_orig_rs),
        }
