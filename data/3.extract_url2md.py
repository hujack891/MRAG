import re
import requests
from bs4 import BeautifulSoup
import html2text
import os
import shutil
import time
import random

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logs.log_config import setup_logging

# create the log system
logger = setup_logging(os.path.splitext(os.path.basename(__file__))[0])

# 清理文件夹内的全部文件
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
                shutil.rmtree(file_path)  # 删除文件夹及其内容
        except Exception as e:
            logger.error(f"删除失败: {file_path}，错误信息: {e}")

# extract the link
def extract_links_from_markdown(file_path):
    """
    从markdown文件中提取所有链接的文字和URL
    返回两个列表：文字列表和链接列表
    """
    text_list = []
    url_list = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 使用手动解析方法来正确处理URL中包含括号的情况
        i = 0  # 当前查找位置
        while i < len(content):
            # 寻找 [
            start_bracket = content.find('[', i)  # 找到下一个[左方括号的索引
            if start_bracket == -1:  # 如果没有找到返回-1
                break
            
            # 寻找对应的 ]
            end_bracket = content.find(']', start_bracket)
            if end_bracket == -1:
                i = start_bracket + 1
                continue
            
            # 检查 ] 后面是否紧跟 (
            if end_bracket + 1 >= len(content) or content[end_bracket + 1] != '(':
                i = start_bracket + 1
                continue
            
            # 提取链接文本
            link_text = content[start_bracket + 1:end_bracket]
            
            # 寻找匹配的 )，需要考虑括号平衡
            paren_start = end_bracket + 2  # 跳过 ](
            paren_count = 1
            j = paren_start
            
            while j < len(content) and paren_count > 0:
                if content[j] == '(':
                    paren_count += 1
                elif content[j] == ')':
                    paren_count -= 1
                j += 1
            
            if paren_count == 0:
                # 找到了匹配的 )
                url = content[paren_start:j-1]
                text_list.append(link_text)
                url_list.append(url)
                i = j
            else:
                i = start_bracket + 1
    
    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
    except Exception as e:
        logger.error(f"读取文件时出错: {e}")
    
    return text_list, url_list

# convert URLs to Markdown format
def url_to_markdown(url):
    logger.debug(f"开始处理URL: {url}")
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        )
    }

    try:
        # 请求网页内容
        logger.debug(f"发送HTTP请求到: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        logger.debug(f"HTTP响应状态码: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"请求失败，状态码: {response.status_code}, URL: {url}")
            return None


        print(response.text)
        # 解析网页结构
        soup = BeautifulSoup(response.text, 'html.parser')
        for img in soup.find_all("img"):
            data_src = img.get("data-src")
            if data_src:
                img["src"] = data_src


        logger.debug(f"网页标题: {soup.title.string if soup.title else 'No title'}")

        # 提取文章主内容区域
        main_content = soup.find("div", class_="content") or soup.body
        
        if not main_content:
            logger.error(f"未找到主要内容区域, URL: {url}")
            logger.debug(f"网页结构: {soup.prettify()[:500]}...")
            return None

        logger.debug(f"找到主内容区域，长度: {len(str(main_content))}")

        
        # 转换为 Markdown
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.body_width = 0  # 不换行
        markdown = h.handle(str(main_content))  # 将 HTML 内容（如网页中的 DOM 结构）转换为 Markdown 格式的纯文本。

        result = markdown.strip()
        logger.debug(f"生成Markdown长度: {len(result)}")
        
        if not result:
            logger.warning(f"转换后的Markdown内容为空, URL: {url}")
        
        return result
        
    except requests.exceptions.Timeout:
        logger.error(f"请求超时, URL: {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"请求异常: {e}, URL: {url}")
        return None
    except Exception as e:
        logger.error(f"处理URL时发生未知错误: {e}, URL: {url}")
        return None

# 删除只包含'Play'的行
def remove_play_lines(content):
    """删除只包含'Play'的行"""
    lines = content.split('\n')
    # 过滤掉只包含'Play'的行（去除前后空白后判断）
    filtered_lines = [line for line in lines if line.strip() != 'Play']
    return '\n'.join(filtered_lines)

# Strip hyperlinks from the text
def remove_text_links(content):
    """删除文本链接的简化处理：
    1. 匹配到文本链接后将括号内容删掉
    2. 将[]去掉，并且添加一个空格
    3. 检测到4个连续*，则直接删掉
    **Equip your first Gourd**[**Soak**](https://www.ign.com/wikis/black-myth-wukong/Soaks "Soaks")**.**
    **Equip your first Gourd**[**Soak**]**.**
    **Equip your first Gourd** **Soak****.**
    **Equip your first Gourd** **Soak.**
    """
    
    # 第1步：删除文本链接中的括号内容，但保留图片链接
    # 匹配 [text](url) 但不匹配 ![text](url)
    def remove_url_part(match):
        # 检查是否是图片链接（前面有!）
        start_pos = match.start()
        if start_pos > 0 and content[start_pos - 1] == '!':
            return match.group(0)  # 保留图片链接
        
        # 对于文本链接，只保留 [text] 部分
        return match.group(1)  # 返回 [text] 部分
    
    # 步骤1：删除括号部分 [text](url) -> [text]
    link_pattern = re.compile(r'(\[[^\[\]]*?\])\([^)]*?\)')
    content = link_pattern.sub(remove_url_part, content)
    
    # 第2步：删除[]并添加空格，但要保护图片链接
    # 匹配 [text] 替换为 " text"，但要检查前面是否已经有空格，以及是否是图片链接
    def replace_brackets(match):
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
    content = bracket_pattern.sub(replace_brackets, content)
    
    # 第3步：删除4个连续的星号
    content = re.sub(r'\*{4,}', '', content)
    
    # 额外清理：删除3个连续星号变为2个（保持粗体格式）
    content = re.sub(r'\*{3}', '**', content)
    
    return content

# Remove image links from Markdown
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

# Normalize filename
def clean_filename(filename: str) -> str:
    """
    清理文件名，将不适合作为文件名的字符替换为下划线
    
    参数:
        filename: 原始文件名
    返回:
        清理后的文件名
    """
    # 先处理连字符两边的空格，直接删除而不是替换为下划线
    # 例如: "Chapter 2 - Yellow Sand" -> "Chapter 2-Yellow Sand"
    clean_name = re.sub(r'\s*-\s*', '-', filename)
    
    # Windows和Linux都不允许的字符：< > : " | ? * \ /
    # 对于剩余的空格，替换为下划线
    invalid_chars = r'[<>:"|?*\\/\s]'
    
    # 将无效字符替换为下划线
    clean_name = re.sub(invalid_chars, '_', clean_name)
    
    # 移除多个连续的下划线，替换为单个下划线
    clean_name = re.sub(r'_{2,}', '_', clean_name)
    
    # 移除开头和结尾的下划线
    clean_name = clean_name.strip('_')
    
    # 如果清理后的文件名为空，使用默认名称
    if not clean_name:
        clean_name = "unnamed_file"
    
    return clean_name

def main():
    handle_web_num = 0  # The total number of processed documents
    logger.info("start!")
    output_folder = "./data/doc" # Output md file

    os.makedirs(output_folder, exist_ok=True)

    logger.info(f"Start cleaning folder: {output_folder}")
    clean_folder(output_folder)
    logger.info(f"Folder cleaned: {output_folder}")

    md_file_path = "./data/ign_sidebar_bk.md"
    texts, urls = extract_links_from_markdown(md_file_path)

    for url, text in zip(urls, texts):
        if not url.strip():
            logger.error(f"跳过空链接: {text}")
            continue
        handle_web_num += 1
        title = f"# {text}\n\n"
        markdown_text = url_to_markdown(url)

        time.sleep(random.uniform(1, 2))  # Random delay to avoid being blocked

        if isinstance(markdown_text, str) and markdown_text.strip():  # 判断变量 markdown_text 是否为一个非空字符串
            markdown_content = title + markdown_text
            
            # Data Cleaning
            # Delete "Play"
            markdown_content = remove_play_lines(markdown_content)
            # 文字链接直接删除
            markdown_content = remove_text_links(markdown_content)
            # 图片链接进行清理
            markdown_content = clean_markdown_image_links(markdown_content)
            # 去除横线
            markdown_content = remove_markdown_hr(markdown_content)
            # 删除多余的空行
            markdown_content = re.sub(r'\n\s*\n\s*\n', '\n\n', markdown_content)
            # 保存到文件夹中
            clean_text = clean_filename(text)
            file_name = os.path.join(output_folder, f"{clean_text}.md")
            try:
                with open(file_name, "w", encoding="utf-8") as f:
                    f.write(markdown_content)
                logger.debug(f"已保存: {file_name}")
            except Exception as e:
                logger.error(f"写入文件失败: {file_name}, 错误: {e}")
        else:
            logger.error(f"URL 内容为空或解析失败，跳过：{url}")

    logger.info(f"Processed {handle_web_num} links in total")
    logger.info("over!")

if __name__ == "__main__":
    main()
