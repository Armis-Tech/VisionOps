"""Download standard YOLO11n COCO weights into the project root.
Ultralytics handles model download when the named weights are unavailable.
"""
from pathlib import Path
from ultralytics import YOLO

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    weights = root / 'yolo11n.pt'
    YOLO(str(root / 'yolo11n.pt') if weights.exists() else 'yolo11n.pt')
    if not weights.exists():
        print('Model was loaded, but the weights are not in the project root.')
        print('Copy yolo11n.pt from the current working directory to:', root)
    else:
        print('Ready:', weights)
