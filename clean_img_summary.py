import os
import json
import re

# 定义处理文件夹路径
chunks_folder = r'g:\hu\Desktop\task\RAG\index\img_summary\withcontext\chunks'

# 定义正则表达式模式来匹配需要清理的部分
patterns = [
    # 匹配 "### Preceding Text Summary:" 或 "**Preceding Text Summary:**"
    (r'(#+|\*\*) *Preceding Text Summary:( *\*\*)?', 'Preceding Text Summary:'),
    # 匹配 "### Image Summary:" 或 "**Image Summary:**"
    (r'(#+|\*\*) *Image Summary:( *\*\*)?', 'Image Summary:'),
    # 匹配 "### Following Text Summary:" 或 "**Following Text Summary:**"
    (r'(#+|\*\*) *Following Text Summary:( *\*\*)?', 'Following Text Summary:')
]

def clean_summary(text):
    """清理摘要文本，移除特殊标记"""
    if not text:
        return text
    
    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    
    return result

def process_json_files():
    """处理文件夹中的所有JSON文件"""
    # 获取文件夹中的所有JSON文件
    json_files = [f for f in os.listdir(chunks_folder) if f.endswith('.json')]
    total_files = len(json_files)
    processed_files = 0
    
    print(f"找到 {total_files} 个JSON文件需要处理")
    
    for filename in json_files:
        file_path = os.path.join(chunks_folder, filename)
        
        try:
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否有img_summary字段
            if 'img_summary' in data:
                # 清理img_summary内容
                data['img_summary'] = clean_summary(data['img_summary'])
                
                # 写回文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                
                processed_files += 1
                if processed_files % 100 == 0:
                    print(f"已处理 {processed_files}/{total_files} 个文件")
        
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
    
    print(f"处理完成! 共处理了 {processed_files}/{total_files} 个文件")

if __name__ == "__main__":
    process_json_files()