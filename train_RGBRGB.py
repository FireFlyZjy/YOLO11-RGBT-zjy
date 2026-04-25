import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/26-RGBT/yolo26-RGBRGB6C-midfusion.yaml').load('weights/yolo26n.pt')  # 只是将yaml里面的 ch设置成 6 ,红外部分改为 SilenceChannel, [ 3,6 ] 即可
    # model.load(r'yolo11n-RGBRGB6C-midfussion.pt') # loading pretrain weights 网盘下载
    model.train(data=R'ultralytics/cfg/datasets/LLVIP_zjy.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=4,
                close_mosaic=0,
                workers=0,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBRGB6C",
                channels=6,  #
                project=r'C:\Users\Patrick\Desktop\DeepLearning\Code\YOLOv11to26-RGBT\runs\LLVIP\26dual-demo',
                name='yolo26n-RGBRGB6C-midfusion',
                # val=True,
                )