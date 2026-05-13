from ultralytics import YOLO

model = YOLO("runs/zjy_train/LLVIP/yolo11n-LLVIP_6C-latefusion/weights/best.pt")
model.val(
    data="ultralytics/cfg/datasets/LLVIP_zjy.yaml",
    workers=0,
    device='0',
    batch=4,
    use_simotm="RGBRGB6C",
    channels=6,  #
    project='runs/zjy_val/LLVIP',
    name='yolo11n-LLVIP_6C-latefusion',
)
