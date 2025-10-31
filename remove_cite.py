import re
import os

def remove_cite_tags(input_file):
    # 读取 Markdown 文件内容
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # 使用正则表达式匹配 [cite: ...] 形式的内容
    cleaned_text = re.sub(r'\[cite:[^\]]*\]', '', text)

    # 生成输出文件路径（例如 input.md → input_removed.md）
    base, ext = os.path.splitext(input_file)
    output_path = f"{base}_removed{ext}"

    # 写入新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    print(f"✅ 已处理完成，保存到: {output_path}")

# 示例用法：
remove_cite_tags("/Users/hovsco/Documents/CUHKSZ/2025_Fall/CSC4005/CUHKSZ-CSC4005/project2/report.md")
