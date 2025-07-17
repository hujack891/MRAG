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
OUTPUT_DIR = "./index/img_summary/v3"
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
    chunk_id = 0

    h1_buffer = []
    h2_buffer = []
    h3_buffer = []

    img_below_buffer = []
    in_image_section = False

    for line in lines:
        stripped = line.strip()

        # 一级标题
        if stripped.startswith("# "):
            current_h1 = stripped[2:].strip()
            current_h2 = ""
            current_h3 = ""
            h1_buffer = [stripped]
            h2_buffer = []
            h3_buffer = []
            in_image_section = False

        # 二级标题
        elif stripped.startswith("## "):
            current_h2 = stripped[3:].strip()
            current_h3 = ""
            h2_buffer = [stripped]
            h3_buffer = []
            in_image_section = False

        # 三级标题
        elif stripped.startswith("### "):
            current_h3 = stripped[4:].strip()
            h3_buffer = [stripped]
            in_image_section = False

        # 图片
        elif "![" in stripped and "](" in stripped:
            # 提取图片信息
            alt_text_start = stripped.find("![") + 2
            alt_text_end = stripped.find("]", alt_text_start)
            img_url_start = stripped.find("](") + 2
            img_url_end = stripped.find(")", img_url_start)

            alt_text = stripped[alt_text_start:alt_text_end].strip()
            img_url = stripped[img_url_start:img_url_end].strip()

            # 上文整合逻辑（包含标题 + buffer）
            above_text_parts = []
            if current_h1:
                above_text_parts += h1_buffer
            if current_h2:
                above_text_parts += h2_buffer
            if current_h3:
                above_text_parts += h3_buffer
            above_text_parts += img_below_buffer  # 当前标题下图像上方文字

            # 保存 chunk
            summary_chunks.append(SummaryData(
                chunk_id=0,  # 固定为0，后续处理再编号
                source_file=filename,
                h1_title=current_h1,
                h2_title=current_h2,
                h3_title=current_h3,
                img_url=img_url,
                alt_text=alt_text,
                position_desc=chunk_id,
                img_above_text="\n".join(above_text_parts).strip(),
                img_below_text="",  # 下面再填充
                summary_promot="",
                img_summary=""
            ))

            chunk_id += 1
            in_image_section = True
            img_below_buffer = []

        elif stripped == "":
            # 空行
            if in_image_section:
                img_below_buffer.append("")
        else:
            # 正文内容
            if stripped.startswith("#"):
                continue  # 已处理过标题
            if in_image_section:
                img_below_buffer.append(stripped)
            else:
                if current_h3:
                    h3_buffer.append(stripped)
                elif current_h2:
                    h2_buffer.append(stripped)
                elif current_h1:
                    h1_buffer.append(stripped)

    # 更新最后一个图片的 img_below_text
    if summary_chunks:
        last_chunk = summary_chunks[-1]
        last_chunk.img_below_text = "\n".join(img_below_buffer).strip()

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