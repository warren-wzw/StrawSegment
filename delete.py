import enum
import os
import sys
os.chdir(sys.path[0])

# 设置准确率阈值
accuracy_threshold = 0.91

# 读取准确率数据并存储到哈希表中
accuracy_map = {}
with open('record.txt', 'r') as record_file:
    for line in record_file:
        parts = line.strip().split()
        index = int(parts[1])
        accuracy = float(parts[-1])
        accuracy_map[index] = accuracy

# 定义文件夹路径
src_folder = './dataset/src/val/'
label_folder = './dataset/label/val/'
indices_to_delete = [index for index, accuracy in accuracy_map.items() if accuracy < accuracy_threshold]

# 获取文件名索引（假设文件名格式为 image{index}.png 和 label{index}.png）
def get_index_from_filename(filename):
    # 从文件名中提取索引
    base_name = os.path.splitext(filename)[0]
    return int(base_name.replace('image', '').replace('label', ''))

# 获取 src 和 label 文件夹中的所有文件，并按文件名排序
src_files = sorted(os.listdir(src_folder), key=get_index_from_filename)
label_files = sorted(os.listdir(label_folder), key=get_index_from_filename)

# 删除准确率低于阈值的文件
for index,(src_file, label_file) in enumerate(zip(src_files, label_files)):
    if index in indices_to_delete:        
        if os.path.exists(src_folder+src_file):
            os.remove(src_folder+src_file)
            print(f"Deleted {src_file}")
        if os.path.exists(label_folder+label_file):
            os.remove(label_folder+label_file)
            print(f"Deleted {label_file}")