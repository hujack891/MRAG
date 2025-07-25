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
OUTPUT_DIR = "./index/img_summary/nocontext"
os.makedirs(OUTPUT_DIR, exist_ok=True)  

@dataclass
class SummaryData:
    chunk_id: int
    source_file: str
    img_url: str             
    alt_text: str            
    position_desc: int       
    summary_promot: str       
    img_summary: str          

    def to_serializable_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "img_url": self.img_url,
            "alt_text": self.alt_text,
            "position_desc": self.position_desc,
            "summary_promot": self.summary_promot,
            "img_summary": self.img_summary
        }
    
def extract_chunks_from_markdown(content: str, filename: str) -> List[SummaryData]:
    pattern = r'!\[([^\]]*?)\]\(\s*(<.*?>|[^)\s]+(?:\s[^)\s]+)*)\s*\)'
    matches = list(re.finditer(pattern, content))

    summary_chunks = []
    position = 1

    for match in matches:
        alt_text = match.group(1).strip()
        img_url = match.group(2).strip('<>')

        summary_chunks.append(SummaryData(
            chunk_id=0,
            source_file=filename,
            img_url=img_url,
            alt_text=alt_text,
            position_desc=position,
            summary_promot='',
            img_summary=''
        ))

        position += 1
    return summary_chunks

def build_prompt_text(chunk_data:SummaryData):
    """构造摘要提示词"""
    return f"""
        "You are shown an image extracted from a video game walkthrough or strategy guide.\n\n"
        "Your task is to write a short, clear, and informative **English summary** of the image.\n"
        "Do **not** speculate beyond what is explicitly visible in the image."
        """.strip()  

def deduplicate_data(data_list: List[SummaryData]) -> List[SummaryData]:
    seen = {}
    for item in data_list:
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
        chunk = extract_chunks_from_markdown(content = content,filename = filename)
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