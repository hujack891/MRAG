import os, sys, re, json
import hashlib
from openai import OpenAI
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logs.log_config import setup_logging
from config import BaseConfig, SUMMARYConfig

# 初始化运行日志
logger = setup_logging(os.path.splitext(os.path.basename(__file__))[0])


baseconfig = BaseConfig()
summaryconfig = SUMMARYConfig()
client = OpenAI(
    base_url=baseconfig.SUMMARY_URL,
    api_key=baseconfig.SUMMARY_API
)
INPUT_DIR = "./data/doc_cleaned"  
OUTPUT_DIR = "./index/img_summary/withcontext"
os.makedirs(OUTPUT_DIR, exist_ok=True)  

@dataclass
class SummaryData:
    chunk_id: int
    source_file: str
    h1_title: str            # 当前一级标题
    h2_title: str            # 当前二级标题（可为空）
    h3_title: str            # 当前三级标题（可为空）
    img_url: str             # 图片url
    alt_text: str            # 图片alt文本
    position_desc: int       # 第几张图片
    img_above_text: str      # 图片的上文
    img_below_text: str      # 图片的下文
    summary_promot: str      # 提示词
    img_summary: str         # 图片总结

    def to_serializable_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "h1_title": self.h1_title,         
            "h2_title": self.h2_title,
            "h3_title": self.h3_title,
            "img_url": self.img_url,
            "alt_text": self.alt_text,
            "position_desc": self.position_desc,
            "img_above_text": self.img_above_text,
            "img_below_text": self.img_below_text,
            "summary_promot": self.summary_promot,
            "img_summary": self.img_summary
        }

def extract_chunks_from_markdown(content: str, filename: str) -> List[SummaryData]:
    lines = content.split('\n')
    summary_chunks = []
    
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""
    chunk_id = 1
    
    # 存储各级标题的完整内容
    h1_full_content = []  # 一级标题下的所有内容（不包含子标题）
    h2_full_content = []  # 二级标题下的所有内容（不包含子标题）
    h3_full_content = []  # 三级标题下的所有内容
    
    # 临时缓冲区，用于收集当前标题下到图片前的内容
    current_level_buffer = []
    
    # 当前标题层级（1, 2, 3）
    current_level = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # 一级标题
        if stripped.startswith("# "):
            current_h1 = stripped[2:].strip()
            current_h2 = ""
            current_h3 = ""
            current_level = 1
            
            # 保存之前一级标题的完整内容
            h1_full_content = [stripped]
            h2_full_content = []
            h3_full_content = []
            current_level_buffer = []
            
        # 二级标题
        elif stripped.startswith("## "):
            current_h2 = stripped[3:].strip()
            current_h3 = ""
            current_level = 2
            
            # 将之前收集的内容加入到一级标题的完整内容中
            h1_full_content.extend(current_level_buffer)
            
            # 开始新的二级标题
            h2_full_content = [stripped]
            h3_full_content = []
            current_level_buffer = []
            
        # 三级标题
        elif stripped.startswith("### "):
            current_h3 = stripped[4:].strip()
            current_level = 3
            
            # 将之前收集的内容加入到对应级别的完整内容中
            if current_level == 2:
                h1_full_content.extend(current_level_buffer)
            elif current_level == 3:
                h2_full_content.extend(current_level_buffer)
            
            # 开始新的三级标题
            h3_full_content = [stripped]
            current_level_buffer = []
            
        # 图片
        elif "![" in stripped and "](" in stripped:
            # 提取图片信息
            alt_text_start = stripped.find("![") + 2
            alt_text_end = stripped.find("]", alt_text_start)
            img_url_start = stripped.find("](") + 2
            img_url_end = stripped.find(")", img_url_start)
            
            alt_text = stripped[alt_text_start:alt_text_end].strip()
            img_url = stripped[img_url_start:img_url_end].strip()
            
            # 构建 img_above_text
            above_text_parts = []
            
            if current_level == 1:
                # 一级标题下的图片：包含一级标题 + 一级标题下到图片前的内容
                above_text_parts.extend(h1_full_content)
                above_text_parts.extend(current_level_buffer)
                
            elif current_level == 2:
                # 二级标题下的图片：包含一级标题的完整内容 + 二级标题 + 二级标题下到图片前的内容
                above_text_parts.extend(h1_full_content)
                above_text_parts.extend(h2_full_content)
                above_text_parts.extend(current_level_buffer)
                
            elif current_level == 3:
                # 三级标题下的图片：包含一级标题的完整内容 + 二级标题的完整内容 + 三级标题 + 三级标题下到图片前的内容
                above_text_parts.extend(h1_full_content)
                above_text_parts.extend(h2_full_content)
                above_text_parts.extend(h3_full_content)
                above_text_parts.extend(current_level_buffer)
            
            # 收集图片下方的文本（在当前标题层级范围内）
            below_content = []
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                
                # 检查是否遇到同级或更高级的标题
                is_break = False
                if current_level == 3:
                    is_break = next_line.startswith("### ") or next_line.startswith("## ") or next_line.startswith("# ")
                elif current_level == 2:
                    is_break = next_line.startswith("## ") or next_line.startswith("# ")
                elif current_level == 1:
                    is_break = next_line.startswith("# ")
                
                if is_break:
                    break
                
                # 检查是否遇到下一个图片
                if "![" in next_line and "](" in next_line:
                    break
                
                # 收集内容
                below_content.append(next_line if next_line else "")
            
            # 创建chunk
            summary_chunks.append(SummaryData(
                chunk_id=chunk_id,
                source_file=filename,
                h1_title=current_h1,
                h2_title=current_h2,
                h3_title=current_h3,
                img_url=img_url,
                alt_text=alt_text,
                position_desc=chunk_id,
                img_above_text="\n".join(above_text_parts).strip(),
                img_below_text="\n".join(below_content).strip(),
                summary_promot="",
                img_summary=""
            ))
            
            chunk_id += 1
            current_level_buffer = []  # 清空缓冲区
            
        elif stripped == "":
            # 空行
            current_level_buffer.append("")
            
        else:
            # 正文内容
            if not stripped.startswith("#"):  # 确保不是标题
                current_level_buffer.append(stripped)
    
    return summary_chunks


def build_prompt_text(chunk_data):
    """构造摘要提示词"""

    if chunk_data.h3_title != "":
        return f"""
            You are given an image and related metadata and context.

            Metadata:
            - Document Title: {chunk_data.h1_title}
            - Section Title: {chunk_data.h2_title}
            - Sub-Section Title: {chunk_data.h3_title}

            Context:
            - Preceding Text: "{chunk_data.img_above_text}"
            - Following Text: "{chunk_data.img_below_text}"

            Your task:
            1. Generate a **short, clear, and informative English summary** of the image.
            2. Focus primarily on the **preceding and following text** to infer the image's **purpose, content, or narrative role**.
            3. Incorporate any relevant metadata (e.g., section or document titles) to improve coherence, **but do not assume any domain-specific knowledge** unless explicitly stated.
            4. Do **not** speculate beyond the visible content.\n\n

            Output Format:
            Preceding Text Summary:
            Provide a concise summary of the preceding text that gives context to the image.
            Image Summary:
            Provide a summary of the image, describing its purpose and content based on the context.
            Following Text Summary:
            Provide a concise summary of the following text that elaborates on the information related to the image.
            """.strip()            

    elif chunk_data.h2_title != "":
        return f"""
            You are given an image and related metadata and context.

            Metadata:
            - Document Title: {chunk_data.h1_title}
            - Section Title: {chunk_data.h2_title}

            Context:
            - Preceding Text: "{chunk_data.img_above_text}"
            - Following Text: "{chunk_data.img_below_text}"

            Your task:
            1. Generate a **short, clear, and informative English summary** of the image.
            2. Focus primarily on the **preceding and following text** to infer the image's **purpose, content, or narrative role**.
            3. Incorporate any relevant metadata (e.g., section or document titles) to improve coherence, **but do not assume any domain-specific knowledge** unless explicitly stated.
            4. Do **not** speculate beyond the visible content.\n\n

            Output Format:
            Preceding Text Summary:
            Provide a concise summary of the preceding text that gives context to the image.
            Image Summary:
            Provide a summary of the image, describing its purpose and content based on the context.
            Following Text Summary:
            Provide a concise summary of the following text that elaborates on the information related to the image.

            """.strip()
    else:
        return f"""
            You are given an image and related metadata and context.

            Metadata:
            - Document Title: {chunk_data.h1_title}

            Context:
            - Preceding Text: "{chunk_data.img_above_text}"
            - Following Text: "{chunk_data.img_below_text}"

            Your task:
            1. Generate a **short, clear, and informative English summary** of the image.
            2. Focus primarily on the **preceding and following text** to infer the image's **purpose, content, or narrative role**.
            3. Incorporate any relevant metadata (e.g., section or document titles) to improve coherence, **but do not assume any domain-specific knowledge** unless explicitly stated.
            4. Do **not** speculate beyond the visible content.\n\n
            
            Output Format:
            Preceding Text Summary:
            Provide a concise summary of the preceding text that gives context to the image.
            Image Summary:
            Provide a summary of the image, describing its purpose and content based on the context.
            Following Text Summary:
            Provide a concise summary of the following text that elaborates on the information related to the image.
            """.strip()

def deduplicate_data(data_list: List[SummaryData]) -> List[SummaryData]:
    seen = {}
    for item in data_list:
        # 去除空内容项
        # 若 chunk_id 未出现过，则添加
        if item.chunk_id not in seen:
            seen[item.chunk_id] = item
    return list(seen.values())

def process_chunk(index, chunk_data):
    try:
        # 构建 messages 内容
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": chunk_data.summary_promot
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": chunk_data.img_url,
                            "detail": "high"  # 使用高分辨率图像
                        }
                    }
                ]
            }
        ]
        # 调用 API
        response = client.chat.completions.create(
            model=summaryconfig.SUMMARY_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.3,
            timeout=30
        )

        summary = response.choices[0].message.content.strip()
        # 写入 chunk 内容
        chunk_data.img_summary = summary
        return index, chunk_data

    except Exception as e:
        logger.error(f"生成摘要时出错: {str(e)}")
        return None

def main():

    # 读取全部md文件
    all_md_files = []  
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.md'):
            all_md_files.append(filename)   
    if not all_md_files:
        logger.warning("没有找到需要处理的Markdown文件")
        return
    logger.debug(f'找到{len(all_md_files)}个Markdown文件需要处理')

    # 构建每一个chunk
    all_summary_data = []
    for filename in all_md_files:
        input_path = os.path.join(INPUT_DIR, filename)
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    logger.warning(f"文件为空: {filename}")
                    continue
        except Exception as e:
            logger.error(f"读取文件失败 {filename}: {e}")
            continue
        chunk = extract_chunks_from_markdown(content = content, filename = filename)
        all_summary_data.extend(chunk)
    
    # 将数据结构中的提示词和ID补充完整
    for i in range(len(all_summary_data)):
        # 构建提示词
        prompt = build_prompt_text(all_summary_data[i])
        all_summary_data[i].summary_promot = prompt
        only_text = prompt + all_summary_data[i].img_url
        chunk_hash = hashlib.md5(only_text.encode('utf-8')).hexdigest()
        chunk_id = int(chunk_hash, 16) % (10 ** 12)
        all_summary_data[i].chunk_id = chunk_id

    # 对数据结构列表通过ID去重
    all_summary_data = deduplicate_data(all_summary_data)
    
    logger.info(f'去重后有{len(all_summary_data)}个chunk')
    
    # 生成图像摘要 
    with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i, chunk in enumerate(all_summary_data):
                future = executor.submit(process_chunk, i, chunk)
                futures.append(future)

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    i, updated_chunk = result
                    all_summary_data[i] = updated_chunk
                    logger.info(f"成功为 chunk {i} 生成图像摘要（chunk_id={updated_chunk.chunk_id}）")

    # 创建chunks文件夹
    chunks_dir = os.path.join(OUTPUT_DIR, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    chunk_id_to_path = {}
    # 保存chunk信息到JSON文件

    for i, chunk in enumerate(all_summary_data):
        output_path = os.path.join(chunks_dir, f"chunk_data_{i}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunk.to_serializable_dict(), f, ensure_ascii=False, indent=4)
        chunk_id_to_path[str(chunk.chunk_id)] = f"chunks/chunk_data_{i}.json"
        logger.info(f"chunk信息已保存到 {output_path}")

    mapping_path = os.path.join(OUTPUT_DIR, "chunk_id_to_path.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(chunk_id_to_path, f, ensure_ascii=False, indent=4)

    logger.info(f"chunk_id 映射信息已保存到 {mapping_path}")   

if __name__ == "__main__":
    main()