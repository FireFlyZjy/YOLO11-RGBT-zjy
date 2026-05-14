from ultralytics import YOLO

model = YOLO("runs/FLIR/26dual-test/yolo26n-RGBT-midfusion-ASPP-V1/weights/best.pt")
model.val(
    data="ultralytics/cfg/datasets/flir.yaml",
    workers=0,
    device='0',
    batch=4,
    # use_simotm="RGBRGB6C",
    # channels=6,
    use_simotm="RGBT",
    channels=4,
    project='runs/FLIR/val',
    name='yolo26n-RGBT-midfusion-ASPP-V1',
)
