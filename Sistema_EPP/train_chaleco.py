"""Fine-tuning del modelo Chaleco.pt con fotos de los chalecos reales."""
from pathlib import Path
import yaml
from ultralytics import YOLO

BASE = Path(__file__).parent / "data" / "entrenamiento" / "chaleco"

dataset_cfg = {
    "path": str(BASE),
    "train": "images",
    "val": "images",
    "nc": 2,
    "names": ["vest", "no-vest"],
}
yaml_path = BASE / "dataset.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(dataset_cfg, f, default_flow_style=False)

modelo_base = Path(__file__).parent.parent / "Chaleco.pt"
model = YOLO(str(modelo_base))

results = model.train(
    data=str(yaml_path),
    epochs=80,
    imgsz=640,
    lr0=0.0005,
    lrf=0.1,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
    batch=1,
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=30.0,
    translate=0.2,
    scale=0.4,
    shear=10.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    rect=False,
    save=True,
    save_period=10,
    project=str(BASE / "runs"),
    name="train",
    exist_ok=True,
    pretrained=True,
    freeze=10,
    verbose=True,
    device="cpu",
)

model_path = BASE / "runs" / "train" / "weights" / "best.pt"
print(f"\nModelo fine-tuneado guardado en: {model_path}")

# Copiar al directorio del proyecto
import shutil
dest = Path(__file__).parent.parent / "Chaleco.pt"
shutil.copy2(str(model_path), str(dest))
print(f"Copiado a {dest}")
