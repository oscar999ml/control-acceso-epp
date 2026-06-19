from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

DATA = Path(__file__).parent / "data" / "entrenamiento"
MODELOS = Path(__file__).parent / "data" / "modelos"
MODELOS.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASES_CASCO = sorted(["no_casco", "casco"])       # orden alfabético = ImageFolder
CLASES_CHALECO = sorted(["no_chaleco", "chaleco"])

ETIQUETAS = {
    "casco": "helmet",
    "no_casco": "no-helmet",
    "chaleco": "vest",
    "no_chaleco": "no-vest",
}


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _transform(img: np.ndarray, size: int = 224) -> torch.Tensor:
    rgb = _to_rgb(img)
    h, w = rgb.shape[:2]
    s = min(h, w)
    y = (h - s) // 2
    x = (w - s) // 2
    crop = rgb[y:y+s, x:x+s]
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    tensor = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(tensor)
    return tensor.unsqueeze(0)


def _crear_modelo(num_clases: int) -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_clases)
    return model


class ClasificadorEPP:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.modelo_casco: nn.Module | None = None
        self.modelo_chaleco: nn.Module | None = None

    def _cargar_o_crear(self, ruta: Path, clases: list[str]) -> nn.Module:
        model = _crear_modelo(len(clases))
        if ruta.exists():
            state = torch.load(str(ruta), map_location=self.device, weights_only=True)
            model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def cargar_modelos(self):
        # Clasificador de casco desactivado: muy pocas imágenes negativas → sobreajuste
        self.modelo_casco = None
        if (DATA / "chaleco_clf" / "no_chaleco").exists() and len(list((DATA / "chaleco_clf" / "no_chaleco").glob("*"))) > 0:
            self.modelo_chaleco = self._cargar_o_crear(MODELOS / "chaleco_clf.pt", CLASES_CHALECO)

    def _predecir(self, model: nn.Module, roi: np.ndarray, clases: list[str], umbral: float = 0.5) -> tuple[str, float]:
        if roi is None or roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return clases[0], 0.0
        tensor = _transform(roi).to(self.device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            prob, idx = probs.max(dim=1)
        clase = clases[idx.item()]
        conf = prob.item()
        if conf < umbral:
            return clases[0], conf
        return clase, conf

    def predecir_casco(self, roi: np.ndarray) -> tuple[str, float]:
        if self.modelo_casco is None:
            return "no-helmet", 0.5
        clase, conf = self._predecir(self.modelo_casco, roi, CLASES_CASCO)
        return ETIQUETAS[clase], conf

    def predecir_chaleco(self, roi: np.ndarray) -> tuple[str, float]:
        if self.modelo_chaleco is None:
            return "no-vest", 0.5
        clase, conf = self._predecir(self.modelo_chaleco, roi, CLASES_CHALECO)
        return ETIQUETAS[clase], conf

    def entrenar_casco(self, epochs: int = 80, lr: float = 0.0003):
        casco_path = DATA / "casco_clf"
        no_class = casco_path / "no_casco"
        if not no_class.exists() or len(list(no_class.glob("*"))) == 0:
            print("  [SKIP] no_casco vacío — omitiendo clasificador de casco, se usará YOLO")
            return
        self._entrenar(MODELOS / "casco_clf.pt", casco_path, CLASES_CASCO, epochs, lr)

    def entrenar_chaleco(self, epochs: int = 80, lr: float = 0.0003):
        chaleco_path = DATA / "chaleco_clf"
        no_class = chaleco_path / "no_chaleco"
        if not no_class.exists() or len(list(no_class.glob("*"))) == 0:
            print("  [SKIP] no_chaleco vacío — entrenar después cuando tengas negativos")
            return
        self._entrenar(MODELOS / "chaleco_clf.pt", chaleco_path, CLASES_CHALECO, epochs, lr)

    def _entrenar(self, ruta_modelo: Path, data_dir: Path, clases: list[str], epochs: int, lr: float):
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from torchvision.datasets import ImageFolder

        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        full = ImageFolder(str(data_dir), transform=train_transforms)
        n = len(full)
        if n < 2:
            print(f"  [SKIP] solo {n} imágenes, necesitas al menos 2")
            return

        # Use all data for training, report train accuracy only
        # With such small datasets, validation doesn't make sense
        loader = DataLoader(full, batch_size=4, shuffle=True)

        model = _crear_modelo(len(clases))
        model.to(self.device)

        # Class weights for imbalance
        targets = [full.targets[i] for i in range(n)]
        counts = [targets.count(c) for c in range(len(clases))]
        weights = [max(counts) / max(c, 1) for c in counts]
        class_weights = torch.tensor(weights, dtype=torch.float).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_loss = float("inf")
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

            avg_loss = total_loss / len(loader)
            acc = correct / total * 100.0

            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"  [{ruta_modelo.stem}] ep {epoch+1}/{epochs} loss={avg_loss:.4f} acc={acc:.1f}%")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), str(ruta_modelo))

        print(f"  [OK] {ruta_modelo.stem} entrenado, mejor loss: {best_loss:.4f}")
