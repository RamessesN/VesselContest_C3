import os
import torch
from ultralytics import YOLO

def evaluate_yolov8_on_test_set():
    """
    在测试集上评估YOLOv8模型性能。
    """
    # 数据集根路径
    dataset_root = r"/root/autodl-tmp/yolo"
    data_yaml_path = os.path.join(dataset_root, 'data.yaml')

    # 训练好的模型权重路径
    # 请根据实际情况修改此路径
    model_weights_path = r'/root/autodl-tmp/yolo/runs/detect/seaships_yolov8s_initial_run3/weights/best.pt'

    # 加载模型
    try:
        model = YOLO(model_weights_path)
    except Exception as e:
        print(f"错误: 无法加载模型 {model_weights_path}. {e}")
        return

    # 设置设备
    device = '0' if torch.cuda.is_available() else 'cpu'

    # 执行评估
    metrics = model.val(
        data=data_yaml_path,
        split='test',  # 在测试集上评估
        imgsz=640,     # 图像尺寸与训练保持一致
        batch=16,      # 批处理大小
        name='evaluation_on_test_set',  # 评估结果保存目录名
        device=device,
    )

    # 打印关键评估指标
    print(f"测试集 mAP50-95: {metrics.box.map:.4f}")
    print(f"测试集 mAP50: {metrics.box.map50:.4f}")
    print(f"测试集 Precision: {metrics.box.p.item():.4f}")
    print(f"测试集 Recall: {metrics.box.r.item():.4f}")

    # 评估结果保存路径
    results_save_dir = metrics.save_dir
    print(f"所有评估结果已保存至: {results_save_dir}")
    print(f"详细指标CSV: {os.path.join(results_save_dir, 'results.csv')}")
    print(f"评估曲线图: {os.path.join(results_save_dir, 'results.png')}")

if __name__ == '__main__':
    evaluate_yolov8_on_test_set()
