import os
from ultralytics import YOLO

def train_yolov8_model():
    """
    使用YOLOv8训练目标检测模型。
    """
    # 加载预训练模型
    model = YOLO('yolov8s.pt')

    # 数据集配置文件路径
    # 请根据实际情况修改此路径
    data_yaml_path = r'E:\下载\SeaShips(7000)\data.yaml'

    # 定义训练参数
    epochs = 150  # 训练轮次
    img_size = 640 # 输入图像尺寸
    batch_size = 16 # 批处理大小
    run_name = 'seaships_yolov8s_initial_run' # 训练结果保存目录名

    # 设置设备
    # '0' 表示使用第一个GPU，'cpu' 表示使用CPU
    device = '0'

    # 开始训练
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name=run_name,
        device=device,
    )

    # 训练完成提示
    # 结果将保存到 runs/detect/{run_name} 目录下
    print(f"训练完成！结果已保存至 runs/detect/{run_name}")

if __name__ == '__main__':
    train_yolov8_model()
