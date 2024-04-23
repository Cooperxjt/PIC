# 将csvtotal中文件，随机按照40-360移动到test和prior
import os
import random
import shutil

# 定义源文件夹和目标文件夹的路径
source_folder = '/home/zhangshuo/pic/datasets/csv_total'
destination_folder_copy = '/home/zhangshuo/pic/datasets/test/total'
destination_folder_move = '/home/zhangshuo/pic/datasets/prior/total'

# 检查目标文件夹是否存在，如果不存在，则创建
if not os.path.exists(destination_folder_copy):
    os.makedirs(destination_folder_copy)
if not os.path.exists(destination_folder_move):
    os.makedirs(destination_folder_move)

# 获取源文件夹中所有文件的列表
files = [file for file in os.listdir(source_folder) if os.path.isfile(
    os.path.join(source_folder, file))]

# 确保文件总数足够分配
if len(files) >= 40:
    # 随机选择40个文件进行复制
    files_to_copy = random.sample(files, 40)
    for file in files_to_copy:
        shutil.copy(os.path.join(source_folder, file), destination_folder_copy)
        files.remove(file)  # 从列表中移除已复制的文件

    # 将剩余的文件复制到另一个文件夹
    for file in files:
        shutil.copy(os.path.join(source_folder, file), destination_folder_move)
else:
    print("Not enough files in the source folder to select 40 files.")
