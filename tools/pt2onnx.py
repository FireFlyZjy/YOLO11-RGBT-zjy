import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def pt2onnx(pt_path, imgsz=640, simplify=True, opset=None, dynamic=False, half=False, batch=1):
    model = YOLO(str(pt_path))

    # Detect input channels from the loaded model's yaml config
    ch = 3
    if hasattr(model.model, 'yaml') and model.model.yaml:
        ch = model.model.yaml.get('ch', 3)

    export_kwargs = {
        'format': 'onnx',
        'imgsz': imgsz,
        'simplify': simplify,
        'batch': batch,
    }
    if ch != 3:
        export_kwargs['channels'] = ch
    if opset is not None:
        export_kwargs['opset'] = opset
    if dynamic:
        export_kwargs['dynamic'] = True
    if half:
        export_kwargs['half'] = True

    print(f'Exporting {pt_path} -> ONNX (ch={ch}, imgsz={imgsz}, simplify={simplify})')
    result = model.export(**export_kwargs)
    print(f'Done: {result}')
    return result


if __name__ == '__main__':
    pt_path = 'runs/FLIR/26dual-test/yolo26n-RGBT-midfusion-ASPP-V1/weights/best.pt'
    if len(sys.argv) > 1:
        pt_path = sys.argv[1]

    pt2onnx(pt_path, imgsz=640, simplify=True)
