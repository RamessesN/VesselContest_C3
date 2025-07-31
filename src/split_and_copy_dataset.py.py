import os
import random
import shutil


def copy_files_to_yolo_structure(src_image_dir, src_label_dir, dst_base_dir, filenames, image_ext, label_ext):
    """将指定文件复制到YOLO格式的目标结构中。"""
    dst_image_sub_dir = os.path.join(dst_base_dir, 'images')
    dst_label_sub_dir = os.path.join(dst_base_dir, 'labels')

    os.makedirs(dst_image_sub_dir, exist_ok=True)
    os.makedirs(dst_label_sub_dir, exist_ok=True)

    for filename in filenames:
        src_image_path = os.path.join(src_image_dir, filename + image_ext)
        dst_image_path = os.path.join(dst_image_sub_dir, filename + image_ext)

        src_label_path = os.path.join(src_label_dir, filename + label_ext)
        dst_label_path = os.path.join(dst_label_sub_dir, filename + label_ext)

        if os.path.exists(src_image_path):
            shutil.copy(src_image_path, dst_image_path)
        # else: 可以添加警告日志，但此处按要求精简

        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dst_label_path)
        # else: 可以添加警告日志，但此处按要求精简


def split_and_copy_dataset(original_image_dir, original_label_dir, output_root_dir, image_ext, label_ext,
                           train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    """
    将数据集随机划分为训练集、验证集和测试集，并复制文件。
    """
    # 归一化划分比例
    total_ratio = train_ratio + valid_ratio + test_ratio
    train_ratio /= total_ratio
    valid_ratio /= total_ratio
    test_ratio /= total_ratio

    # 获取所有图像文件名
    image_filenames = [os.path.splitext(f)[0] for f in os.listdir(original_image_dir) if f.endswith(image_ext)]

    if not image_filenames:
        print(f"错误: 未找到 '{image_ext}' 类型的图像文件，请检查路径。")
        return

    random.shuffle(image_filenames)

    # 计算划分数量
    total_count = len(image_filenames)
    train_count = int(total_count * train_ratio)
    valid_count = int(total_count * valid_ratio)
    test_count = total_count - train_count - valid_count

    # 定义输出目录
    train_set_base_dir = os.path.join(output_root_dir, 'train')
    valid_set_base_dir = os.path.join(output_root_dir, 'valid')
    test_set_base_dir = os.path.join(output_root_dir, 'test')

    # 复制文件到对应文件夹
    copy_files_to_yolo_structure(
        original_image_dir, original_label_dir, train_set_base_dir,
        image_filenames[:train_count], image_ext, label_ext)

    copy_files_to_yolo_structure(
        original_image_dir, original_label_dir, valid_set_base_dir,
        image_filenames[train_count:train_count + valid_count], image_ext, label_ext)

    copy_files_to_yolo_structure(
        original_image_dir, original_label_dir, test_set_base_dir,
        image_filenames[train_count + valid_count:], image_ext, label_ext)


if __name__ == '__main__':
    # 原始图片文件目录
    original_images = r"E:\下载\SeaShips(7000)\JPEGImages"
    # 转换后的YOLO格式TXT标签文件目录
    original_labels = r'E:\下载\SeaShips(7000)\YOLO_Labels'
    # 划分后数据集的输出根目录
    output_dataset_root = r'E:\下载\SeaShips(7000)'

    # 文件扩展名
    image_ext = '.jpg'
    label_ext = '.txt'

    # 运行数据集划分脚本
    split_and_copy_dataset(
        original_image_dir=original_images,
        original_label_dir=original_labels,
        output_root_dir=output_dataset_root,
        image_ext=image_ext,
        label_ext=label_ext,
        train_ratio=0.7,
        valid_ratio=0.15,
        test_ratio=0.15
    )
