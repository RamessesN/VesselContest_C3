import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

def draw_boxes_on_image(image_path, boxes, output_path):
    """
    在图像上绘制目标框，并保存到指定路径
    :param image_path: 原始图像路径
    :param boxes: 目标框列表，每个框为 (xmin, ymin, xmax, ymax)
    :param output_path: 绘制后的图像保存路径
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
    
    image.save(output_path)  # 保存到指定路径

def parse_xml(xml_path):
    """
    解析 XML 文件，提取目标框信息
    :param xml_path: XML 文件路径
    :return: 目标框列表
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')  # 找到 <bndbox> 标签
        if bndbox is not None:
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            boxes.append((xmin, ymin, xmax, ymax))
    
    return boxes

def process_folders(annotations_folder, images_folder, output_folder):
    """
    遍历 Annotations 文件夹，处理 XML 文件，并在对应的 JPEGImages 文件夹中的 JPG 图像上绘制目标框，
    最后将结果保存到 output 文件夹中
    :param annotations_folder: Annotations 文件夹路径
    :param images_folder: JPEGImages 文件夹路径
    :param output_folder: 输出文件夹路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(annotations_folder):
        if filename.endswith('.xml'):
            xml_path = os.path.join(annotations_folder, filename)
            jpg_filename = filename.replace('.xml', '.jpg')
            jpg_path = os.path.join(images_folder, jpg_filename)
            
            if os.path.exists(jpg_path):
                boxes = parse_xml(xml_path)
                output_path = os.path.join(output_folder, jpg_filename)  # 输出路径
                draw_boxes_on_image(jpg_path, boxes, output_path)
                print(f"Processed {jpg_filename} -> {output_path}")
            else:
                print(f"Skipped {filename}: No corresponding JPG file found")

# 示例用法
annotations_folder = r"E:\下载\WaterScenes-main\data\detection\xml"  # 替换为 Annotations 文件夹路径
images_folder = r"E:\下载\WaterScenes-main\data\image"  # 替换为 JPEGImages 文件夹路径
output_folder = r"E:\下载\WaterScenes-main\data\out"  # 替换为 output 文件夹路径
process_folders(annotations_folder, images_folder, output_folder)