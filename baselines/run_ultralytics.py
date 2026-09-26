"""
在 Ultralytics 环境里训练一个 YOLO 模型 —— 由 train_baselines.py 调用，不需要手动运行。

从模型 yaml 构建（不加载 COCO 预训练权重），除 epochs / batch / imgsz / seed / workers /
device 外全部用 Ultralytics 官方默认值（优化器、学习率、增强、close_mosaic=10、
按名义 batch 64 累积梯度等），评估也用 Ultralytics 自带的 val。
"""
import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="yolov8n.yaml / yolo11n.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    from ultralytics import YOLO

    last = Path(a.project) / a.name / "weights" / "last.pt"
    if last.exists():
        print(f"[run_ultralytics] 从 {last} 续训")
        YOLO(str(last)).train(resume=True)
        return
    YOLO(a.model).train(data=a.data, epochs=a.epochs, batch=a.batch, imgsz=a.imgsz,
                        pretrained=False, seed=a.seed, workers=a.workers, device=a.device,
                        project=a.project, name=a.name, exist_ok=True)


if __name__ == "__main__":
    main()
