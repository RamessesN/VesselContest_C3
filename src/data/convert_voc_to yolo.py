import os
import xml.etree.ElementTree as ET
from PIL import Image
import shutil

def convert_voc_xml_to_yolo_txt(xml_dir, image_dir, output_yolo_label_dir, class_list_path='classes.txt'):
    """
    将 PASCAL VOC 格式的 XML 标注文件转换为 YOLO .txt 格式。
    """
    os.makedirs(output_yolo_label_dir, exist_ok=True)

    # 加载类别名称并创建ID映射
    try:
        with open(class_list_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        class_to_id = {name: i for i, name in enumerate(classes)}
    except FileNotFoundError:
        print(f"错误: 类别列表文件 '{class_list_path}' 未找到。")
        return

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

    # 遍历每个XML文件
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 获取图像文件名
            img_filename = root.find('filename').text
            actual_image_path = os.path.join(image_dir, img_filename)

            # 从图像文件获取尺寸
            try:
                with Image.open(actual_image_path) as img:
                    img_width, img_height = img.size
            except FileNotFoundError:
                # 图像文件不存在则跳过当前XML
                continue
            except Exception:
                # 图像文件读取错误则跳过
                continue

            # 输出.txt文件路径
            output_txt_path = os.path.join(output_yolo_label_dir, os.path.splitext(xml_file)[0] + '.txt')

            with open(output_txt_path, 'w') as out_f:
                # 遍历XML中的每个对象
                for obj in root.findall('object'):
                    obj_name = obj.find('name').text

                    if obj_name not in class_to_id:
                        # 类别不在列表中则跳过此对象
                        continue

                    class_id = class_to_id[obj_name]

                    # 获取边界框坐标
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)

                    # VOC转YOLO归一化格式
                    center_x = ((xmin + xmax) / 2.0) / img_width
                    center_y = ((ymin + ymax) / 2.0) / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height

                    # 写入.txt文件
                    out_f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

        except ET.ParseError:
            # XML解析错误则跳过
            continue
        except Exception:
            # 其他未知错误则跳过
            continue


if __name__ == '__main__':
    # 原始XML标注文件目录
    xml_annotations_dir = r"E:\下载\SeaShips(7000)\Annotations"
    # 对应图像文件目录
    image_files_dir = r"E:\下载\SeaShips(7000)\JPEGImages"
    # 转换后YOLO标签输出目录
    yolo_labels_output_dir = r'E:\下载\SeaShips(7000)\YOLO_Labels'
    # 类别列表文件路径
    classes_file = 'classes.txt'

    # 运行转换脚本
    convert_voc_xml_to_yolo_txt(
        xml_dir=xml_annotations_dir,
        image_dir=image_files_dir,
        output_yolo_label_dir=yolo_labels_output_dir,
        class_list_path=classes_file
    )
