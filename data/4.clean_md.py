import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logs.log_config import setup_logging

# 创建日志系统
logger = setup_logging(os.path.splitext(os.path.basename(__file__))[0])

def clean_folder(folder_path):
    """
    删除指定文件夹中的所有内容
    """
    if not os.path.exists(folder_path):
        logger.info(f"文件夹不存在: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)  # 删除文件夹及其内容
        except Exception as e:
            logger.error(f"删除失败: {file_path}，错误信息: {e}")

def remove_text_links(content):
    """删除文本链接：
    1. 匹配到文本链接后将括号内容删掉
    2. 将[]去掉，并且添加一个空格
    3. 检测到4个连续*，则直接删掉
    **Equip your first Gourd**[**Soak**](https://www.ign.com/wikis/black-myth-wukong/Soaks "Soaks")**.**
    **Equip your first Gourd**[**Soak**]**.**
    **Equip your first Gourd** **Soak****.**
    **Equip your first Gourd** **Soak.**
    """
    
    # 第1步：删除文本链接中的括号内容，但保留图片链接
    # 使用更智能的方法处理嵌套括号的情况
    def remove_url_part_smart(text, start_pos=0):
        """智能处理链接，正确匹配嵌套括号"""
        result = []
        i = start_pos
        while i < len(text):
            if text[i:i+2] == '![':  # 图片链接，跳过
                # 找到对应的]
                bracket_count = 1
                j = i + 2
                while j < len(text) and bracket_count > 0:
                    if text[j] == '[':
                        bracket_count += 1
                    elif text[j] == ']':
                        bracket_count -= 1
                    j += 1
                if j < len(text) and text[j] == '(':
                    # 找到对应的)
                    paren_count = 1
                    j += 1
                    while j < len(text) and paren_count > 0:
                        if text[j] == '(':
                            paren_count += 1
                        elif text[j] == ')':
                            paren_count -= 1
                        j += 1
                    result.append(text[i:j])
                    i = j
                else:
                    result.append(text[i])
                    i += 1
            elif text[i] == '[':  # 可能的文本链接
                # 找到对应的]
                bracket_count = 1
                j = i + 1
                while j < len(text) and bracket_count > 0:
                    if text[j] == '[':
                        bracket_count += 1
                    elif text[j] == ']':
                        bracket_count -= 1
                    j += 1
                
                if j < len(text) and text[j] == '(':
                    # 这是一个链接，找到对应的)
                    link_text = text[i:j]  # [text] 部分
                    paren_count = 1
                    j += 1
                    while j < len(text) and paren_count > 0:
                        if text[j] == '(':
                            paren_count += 1
                        elif text[j] == ')':
                            paren_count -= 1
                        j += 1
                    # 只保留链接文本部分，去掉括号并添加空格
                    link_text_clean = link_text[1:-1]  # 去掉[]
                    if result and result[-1] not in [' ', '\n', '\t']:
                        result.append(' ')
                    result.append(link_text_clean)
                    i = j
                else:
                    # 不是链接，保留原样
                    result.append(text[i])
                    i += 1
            else:
                result.append(text[i])
                i += 1
        
        return ''.join(result)
    
    # 步骤1：使用智能方法处理链接
    content = remove_url_part_smart(content)
    
    # 第2步：处理剩余的独立方括号（非链接的方括号）
    # 由于第1步已经处理了所有链接，这里只需要处理剩余的独立方括号
    # 但要保护图片链接
    def replace_remaining_brackets(match):
        text = match.group(1)  # 获取括号内的文本
        start_pos = match.start()
        
        # 检查是否是图片链接（前面有!）
        if start_pos > 0 and content[start_pos - 1] == '!':
            return match.group(0)  # 保留图片链接的方括号
        
        # 检查前面是否有空格或换行
        if start_pos > 0:
            prev_char = content[start_pos - 1]
            if prev_char in [' ', '\n', '\t']:
                return text  # 前面已有空格，直接返回文本
            else:
                return ' ' + text  # 前面没有空格，添加空格
        else:
            return text  # 在行首，直接返回文本
    
    # 步骤2：处理剩余的[text] -> " text"，但保护图片链接
    bracket_pattern = re.compile(r'\[([^\[\]]*?)\]')
    content = bracket_pattern.sub(replace_remaining_brackets, content)
    
    # 第3步：删除4个连续的星号
    content = re.sub(r'\*{4,}', '', content)
    
    # 额外清理：删除3个连续星号变为2个（保持粗体格式）
    content = re.sub(r'\*{3}', '**', content)
    
    return content

def clean_markdown_image_links(markdown_text: str) -> str:
    """
    清理 Markdown 中图片链接，把 .jpg 之后的参数全部移除。
    例如：xxx.jpg?width=200 -> xxx.jpg

    参数:
        markdown_text: 原始 Markdown 内容字符串
    返回:
        清理后的 Markdown 内容
    """
    # 匹配 Markdown 图片链接的正则：提取并替换 URL 中的.jpg?xxx 部分为 .jpg
    cleaned_text = re.sub(r'(!\[[^\]]*\]\([^)]+?\.jpg)\?[^)]+\)', r'\1)', markdown_text)
    
    cleaned_text = re.sub(r'(!\[[^\]]*\]\([^)]+?\.png)\?[^)]+\)', r'\1)', cleaned_text)
    return cleaned_text

def remove_markdown_hr(content: str) -> str:
    """
    删除 Markdown 中的横线（水平分割线）。
    支持 ---、***、___ 三种格式，并忽略前后空格。

    参数:
        content (str): Markdown 文本内容

    返回:
        str: 清理后的 Markdown 内容
    """
    lines = content.split('\n')
    cleaned_lines = []

    for line in lines:
        # 清除所有只含 ---、***、___ 的行（可有空格）
        if not re.match(r'^\s*(-{3,}|\*{3,}|_{3,})\s*$', line):
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

def clean_markdown_content(content):
    """
    对markdown内容进行完整的清理
    """
    content = remove_text_links(content)
    content = remove_markdown_hr(content)
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    return content

def process_markdown_files(input_folder, output_folder):
    """
    处理输入文件夹中的所有markdown文件，清理后保存到输出文件夹
    """

    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"开始清理输出文件夹: {output_folder}")
    clean_folder(output_folder)
    logger.info(f"输出文件夹已清理: {output_folder}")
    
    if not os.path.exists(input_folder):
        logger.error(f"输入文件夹不存在: {input_folder}")
        return
    
    processed_count = 0
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.md'):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            
            try:
                with open(input_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                cleaned_content = clean_markdown_content(content)
                
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                
                logger.info(f"已处理: {filename}")
                processed_count += 1
                
            except Exception as e:
                logger.error(f"处理文件失败: {filename}, 错误: {e}")
    
    logger.info(f"总共处理了 {processed_count} 个markdown文件")

def main():
    logger.info("开始清理markdown文件!")
    
    input_folder = "./data/doc"
    output_folder = "./data/doc_cleaned"
    
    process_markdown_files(input_folder, output_folder)
    
    logger.info("markdown文件清理完成!")

if __name__ == "__main__":
    main()