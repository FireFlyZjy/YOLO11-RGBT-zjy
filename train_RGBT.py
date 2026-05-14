import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/26-RGBT/yolo26-RGBT-midfusion-SE.yaml').load('weights/yolo26n.pt')
    # model.info(True,True)
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'ultralytics/cfg/datasets/flir.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=4,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD',  # using SGD
                # lr0=0.002,
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                # pairs_rgb_ir=['visible','infrared'] , # default: ['visible','infrared'] , others: ['rgb', 'ir'],  ['images', 'images_ir'], ['images', 'image']
                use_simotm="RGBT",
                channels=4,
                project=r'C:\Users\Patrick\Desktop\DeepLearning\Code\YOLOv11to26-RGBT\runs\FLIR\26CC_Add-test',
                name='test',
                )