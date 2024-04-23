import os

import pandas as pd

for user in os.listdir('/home/zhangshuo/pic/datasets/csv_total'):

    # 读取上传的CSV文件的路径
    file_path = os.path.join(
        '/home/zhangshuo/pic/datasets/csv_total', user
    )

    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 按图片名称分组，并只保留每个组的前24行
    grouped = df.groupby('name').head(24)

    # 指定处理后的数据保存路径
    output_path = os.path.join(
        '/home/zhangshuo/pic/datasets/csv_total_new', user
    )
    # 保存处理后的DataFrame到新的CSV文件，不包含索引
    grouped.to_csv(output_path, index=False)

    print(f'Processed file saved to: {output_path}')
