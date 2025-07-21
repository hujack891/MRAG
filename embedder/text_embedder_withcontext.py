# 二级标题内容加上一级标题下的内容

import os, sys, re, json
import time
from pathlib import Path
import numpy as np
import faiss
import pickle
import hashlib
import tiktoken
from openai import OpenAI
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logs.log_config import setup_logging
from config import BaseConfig, TextEmbedding3LargeConfig

# 初始化运行日志
logger = setup_logging(os.path.splitext(os.path.basename(__file__))[0])

# 初始化tiktoken编码器
encoding = tiktoken.get_encoding("cl100k_base")

@dataclass
class EmbeddingData:
    chunk_id: int
    source_file: str
    content_type: str   
    h1_title: str
    h2_title: str
    h1_content: str
    h2_content: str
    promot: str
    promot_tokens_num: int

    def to_serializable_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "content_type": self.content_type,
            "h1_title": self.h1_title,
            "h2_title": self.h2_title,
            "h1_content": self.h1_content,
            "h2_content": self.h2_content,
            "promot": self.promot,
            "promot_tokens_num": self.promot_tokens_num
        }
    

# 计算文本的tokens数量
def count_tokens(text : str) -> int:
    try:
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        logger.warning(f"计算token时发生错误: {e}")
        return 0

# 从文档中提取chunk
def extract_chunks_from_markdown(content: str, filename: str) -> List[EmbeddingData]:
    lines = content.split('\n')
    current_section = {
        "h1_title": "",
        "h2_title": "",
        "h1_content": "",
        "h2_content": "",
        }  
    
    sections = []  # 存储所有section的列表

    idx = 0  # 段的索引
    # 读取一级标题的内容
    h1_match = re.match(r'^# (.+)', lines[idx].strip())
    if h1_match:
        current_section['h1_title'] =  h1_match.group(1).strip()
        logger.debug('成功读取第一行的内容')
        idx += 1
    else:
        logger.error(f'{filename}当前文档处理错误，没有一级标题')
        return []

    # 提取一级标题下的正文内容（直到第一个 ##）
    h1_content_lines = []
    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith("## "):
            break
        if line:
            h1_content_lines.append(line)
        idx += 1
    current_section["h1_content"] = '\n'.join(h1_content_lines)

# 遍历所有二级标题
    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith("## "):
            # 找到一个新的二级标题
            h2_title = line[3:].strip()
            idx += 1

            h2_content_lines = []
            while idx < len(lines):
                subline = lines[idx].strip()
                if subline.startswith("## "):  # 下一个二级标题，结束当前块
                    break
                if subline:
                    h2_content_lines.append(subline)
                idx += 1

            # 构建当前 section，并保存
            section = {
                "h1_title": current_section["h1_title"],
                "h1_content": current_section["h1_content"],
                "h2_title": h2_title,
                "h2_content": '\n'.join(h2_content_lines)
            }
            sections.append(section)
        else:
            idx += 1  # 防止死循环
    # 把 sections 转为 EmbeddingData数据结构
    embedding_chunks = []
    for idx, sec in enumerate(sections):
        embedding_chunks.append(
            EmbeddingData(
                chunk_id = 0,
                source_file = filename,
                content_type = "one and two",
                h1_title = sec["h1_title"],
                h2_title = sec["h2_title"],
                h1_content = sec["h1_content"],
                h2_content = sec["h2_content"],
                promot = "",
                promot_tokens_num = 0,
            )
        )
    return embedding_chunks            


def main():
    logger.info("========== 第一步：读取环境变量和路径 ==========")
    
    baseconfig = BaseConfig()
    INPUT_DIR = "./data/doc_cleaned"  
    OUTPUT_DIR = "./index/text/withcontext"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("\n========== 第二步：读取文档构建chunk ==========")

    md_files = []  # 全部的md文件名
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.md'):
            md_files.append(filename)   
    if not md_files:
        logger.warning("没有找到需要处理的Markdown文件")
        return
    logger.debug(f'找到{len(md_files)}个Markdown文件需要处理')

    all_chunk_data = []
    for filename in md_files:
        input_path = os.path.join(INPUT_DIR, filename)
        try:
        # 读取文件内容
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                content = re.sub(r'!\[.*?\]\([^)]*\)', '', content)  # 移除图片链接
                if not content:
                    logger.warning(f"文件为空: {filename}")
                    continue
        except Exception as e:
            logger.error(f"读取文件失败 {filename}: {e}")
            continue

        chunk = extract_chunks_from_markdown(content=content,filename=filename)

        all_chunk_data.extend(chunk)
    logger.info(f"成功读取{len(all_chunk_data)}个chunk")

    logger.info("\n========== 第三步：构建提示词,并且用提示词生成哈希值当作chunk_id ==========")
    for i in range(len(all_chunk_data)):

        prompt = (
            f"This is an excerpt from the file '{all_chunk_data[i].source_file}', "
            f"located in the main section titled '{all_chunk_data[i].h1_title}'. "
            f"The section summary is: {all_chunk_data[i].h1_content} "
            f"This excerpt merges content from both the main and sub-sections. "
            f"The sub-section is titled '{all_chunk_data[i].h2_title}', and the detailed content is as follows:\n"
            f"{all_chunk_data[i].h2_content}"
        )
        all_chunk_data[i].promot = prompt

        chunk_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        chunk_id = int(chunk_hash, 16) % (10 ** 12)  # 保证是 int64 范围
        all_chunk_data[i].chunk_id = chunk_id

    logger.info("\n========== 第四步：计算每一个chunk的tokens数量，如果tokens数量超过8000，则进行处理 ==========")
    for i in range(len(all_chunk_data)):
        all_chunk_data[i].promot_tokens_num = count_tokens(all_chunk_data[i].promot)
        if all_chunk_data[i].promot_tokens_num > 8000:

            logger.warning(f"chunk {i} 的tokens数量超过8000，需要进行处理")
            all_chunk_data[i].content_type = 'only two'
            prompt = (
                f"This is an excerpt from the file '{all_chunk_data[i].source_file}', "
                f"located in the main section titled '{all_chunk_data[i].h1_title}'. "
                f"This excerpt merges content from both the main and sub-sections. "
                f"The sub-section is titled '{all_chunk_data[i].h2_title}', and the detailed content is as follows:\n"
                f"{all_chunk_data[i].h2_content}"
            )
            all_chunk_data[i].promot = prompt

    logger.info("\n========== 第五步：创建向量索引库 ==========")
    
    textembedding3largeconfig = TextEmbedding3LargeConfig()
    index = faiss.IndexFlatL2(textembedding3largeconfig.EMBEDDING_DIM)   # 使用 L2 距离的扁平索引（暴力搜索）
    index = faiss.IndexIDMap(index)  # 支持 add_with_ids

    logger.info("\n========== 第六步：将chunk内容转为向量，并保存到向量索引库中 ==========")

    client = OpenAI(
        base_url=baseconfig.EMBEDDING_URL,
        api_key=baseconfig.EMBEDDING_API
    )
    
    def process_chunk(chunk_data):
        try:
            response = client.embeddings.create(
                input=chunk_data.promot,
                model=textembedding3largeconfig.EMBEDDING_MODEL_NAME
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return chunk_data.chunk_id, embedding
        except Exception as e:
            error_msg = str(e)
            logger.error(f"计算向量时发生错误: {error_msg}")
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in all_chunk_data]
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is not None:
                chunk_id, embedding = result
                index.add_with_ids(embedding.reshape(1, -1), np.array([chunk_id]))
                logger.info(f"成功将chunk {i} 的内容与向量建立索引")

    logger.info("\n========== 第七步：保存chunk信息到JOSN文件 ==========")

    # 创建chunks文件夹
    chunks_dir = os.path.join(OUTPUT_DIR, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    chunk_id_to_path = {}
    # 保存chunk信息到JSON文件
    for i, chunk in enumerate(all_chunk_data):
        output_path = os.path.join(chunks_dir, f"chunk_data_{i}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunk.to_serializable_dict(), f, ensure_ascii=False, indent=4)
        chunk_id_to_path[str(chunk.chunk_id)] = f"chunks/chunk_data_{i}.json"
        logger.info(f"chunk信息已保存到 {output_path}")

    mapping_path = os.path.join(OUTPUT_DIR, "chunk_id_to_path.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(chunk_id_to_path, f, ensure_ascii=False, indent=4)

    logger.info(f"chunk_id 映射信息已保存到 {mapping_path}")



    logger.info("\n========== 第八步：保存向量索引库 ==========")
    # 保存向量索引库
    index_path = os.path.join(OUTPUT_DIR, "text_embedder_index.faiss")
    faiss.write_index(index, index_path)
    logger.info(f"向量索引库已保存到 {index_path}")

    

if __name__ == "__main__":
    # 运行主函数
    main() 