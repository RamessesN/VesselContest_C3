import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


def extract_classes_from_xml(xml_file_path):
    """
    从单个 XML 文件中提取所有对象的类别名称。
    """
    classes_in_file = []
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name_tag = obj.find('name')
            if name_tag is not None and name_tag.text:
                classes_in_file.append(name_tag.text.strip())
    except ET.ParseError as e:
        print(f"Warning: Error parsing XML file '{xml_file_path}': {e}")
    except Exception as e:
        print(f"Warning: An unexpected error occurred while processing '{xml_file_path}': {e}")
    return classes_in_file


def find_all_unique_classes_multithreaded(xml_dir, max_workers=None):
    """
    多线程遍历指定目录下的所有 XML 文件，找出所有独特的类别名称。

    Args:
        xml_dir (str): 包含 XML 标注文件的目录路径。
        max_workers (int, optional): 最大线程数。如果为 None，则默认为 CPU 核心数 * 5。

    Returns:
        set: 包含所有独特类别名称的集合。
        dict: 统计每个类别出现次数的字典。
    """
    xml_files = [os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.endswith('.xml')]
    if not xml_files:
        print(f"Error: No XML files found in '{xml_dir}'. Please check the directory path.")
        return set(), {}

    print(f"Found {len(xml_files)} XML files. Starting multi-threaded processing...")

    unique_classes = set()
    class_counts = defaultdict(int)

    # 使用 ThreadPoolExecutor 进行多线程处理
    # max_workers 可以根据你的CPU核心数和文件I/O情况调整
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_xml = {executor.submit(extract_classes_from_xml, xml_file): xml_file for xml_file in xml_files}

        # 遍历已完成的任务
        for i, future in enumerate(as_completed(future_to_xml)):
            xml_file = future_to_xml[future]
            try:
                classes_in_file = future.result()
                for cls_name in classes_in_file:
                    unique_classes.add(cls_name)
                    class_counts[cls_name] += 1
            except Exception as exc:
                print(f"File '{xml_file}' generated an exception: {exc}")

            # 打印进度
            if (i + 1) % 100 == 0 or (i + 1) == len(xml_files):
                print(f"Processed {i + 1}/{len(xml_files)} files. Found {len(unique_classes)} unique classes so far.",
                      end='\r')
    print("\nProcessing complete.")
    return unique_classes, class_counts


# --- 使用示例 ---
if __name__ == '__main__':
    # ================================================================
    # IMPORTANT: 配置你的XML文件目录路径
    # ================================================================
    your_xml_annotations_directory = r"E:\下载\WaterScenes-main\data\detection\xml"  # 请替换为你的实际路径

    # 调用函数查找所有类别
    all_unique_classes, class_occurrence_counts = find_all_unique_classes_multithreaded(
        xml_dir=your_xml_annotations_directory,
        max_workers=os.cpu_count() * 2  # 可以根据需要调整线程数，通常是CPU核心数的1-2倍
    )

    print("\n--- 发现的所有独特类别 ---")
    if all_unique_classes:
        # 对类别进行排序以便更好地查看
        sorted_classes = sorted(list(all_unique_classes))
        for cls in sorted_classes:
            print(f"- {cls} (出现次数: {class_occurrence_counts[cls]})")

        # 提示如何生成 classes.txt
        print("\n--- 建议 'classes.txt' 内容 ---")
        for cls in sorted_classes:
            print(cls)
    else:
        print("未找到任何类别。请检查路径或XML文件内容。")