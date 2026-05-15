from ultralytics import YOLO

 # 加载断点权重（last.pt）
model = YOLO("runs/FLIR/26dual-test/yolo26s-RGBT-midfusion-Att_Coord/weights/last.pt")

# 继续训练（自动继承上次的优化器、学习率状态）
results = model.train(
    resume=True,  # 关键参数：启用续训
    epochs=100,   # 总训练轮次（可大于上次，如上次训了50轮，这里设100则再训50轮）
    batch=4,     # 可修改批次大小（不影响续训）
    device='0',
    # lr0=0.0001    # 可调整学习率（若觉得上次学习率过高/过低）
    project=r'runs/FLIR/26dual-test',
    name='yolo26s-RGBT-midfusion-Att_Coord',
)

