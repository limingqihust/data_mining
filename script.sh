#!/bin/bash

# 指定存放 .tgz 文件的目录
tgz_directory="/home/Data-7T-nvme/lzq/tencent_blk_trace/"
# 指定解压目标目录
extract_directory="/home/Data-7T-nvme/lzq/tencent_blk_trace/"  # 替换为实际的目标路径

# 创建解压目录（如果不存在）
mkdir -p "$extract_directory"

# 遍历所有 .tgz 文件并解压
for file in "$tgz_directory"*.tgz; do
    if [ -f "$file" ]; then  # 确保这是一个文件
        echo "正在解压: $file"
        tar -xvzf "$file" -C "$extract_directory"
    else
        echo "没有找到 .tgz 文件"
    fi
done

echo "解压完成。"
