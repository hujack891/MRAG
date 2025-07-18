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

# 读取环境变量和路径 
baseconfig = BaseConfig()
INPUT_DIR = "./index/img_summary/v3/chunks"  
OUTPUT_DIR = "./index/image/v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = OpenAI(
    base_url=baseconfig.EMBEDDING_URL,
    api_key=baseconfig.EMBEDDING_API
)
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
    summary_promot: str      # 暂为空
    img_summary: str         # 图片总结
    embedding_prompt: str    # 图片总结的prompt
    generate_prompt: str     # 检索到该chunk后，提供给AI生成内容的信息

    def to_serializable_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "h1_title": self.h1_title,         # 可留空或使用默认标识
            "h2_title": self.h2_title,
            "h3_title": self.h3_title,
            "img_url": self.img_url,
            "alt_text": self.alt_text,
            "position_desc": self.position_desc,
            "img_above_text": self.img_above_text,
            "img_below_text": self.img_below_text,
            "summary_promot": self.summary_promot,
            "img_summary": self.img_summary,
            "embedding_prompt": self.embedding_prompt,
            "generate_prompt": self.generate_prompt
        }

def load_chunks_to_chunk_data(file_path: str) -> List[SummaryData]:
    
    # 读取文件夹下的全部json文件
    json_files = []  # 全部的json文件名
    try:
        for file in os.listdir(file_path):
            if file.endswith(".json"):
                json_files.append(file)
    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"JSON解码错误: {file_path}")
        return []

    all_chunk_data = []
    for file in json_files:
        with open(os.path.join(file_path, file), 'r', encoding='utf-8') as file:
            data = json.load(file)
            chunk_data = SummaryData(
                chunk_id=data.get("chunk_id", 0),
                source_file=data.get("source_file", ""),
                h1_title=data.get("h1_title", ""),
                h2_title=data.get("h2_title", ""),
                h3_title=data.get("h3_title", ""),
                img_url=data.get("img_url", ""),
                alt_text=data.get("alt_text", ""),
                position_desc=data.get("position_desc", 0),
                img_above_text=data.get("img_above_text", ""),
                img_below_text=data.get("img_below_text", ""),
                summary_promot=data.get("summary_promot", ""),
                img_summary=data.get("img_summary", ""),
                embedding_prompt=data.get("embedding_prompt", ""),
                generate_prompt=''
            )
            all_chunk_data.append(chunk_data)
    return all_chunk_data

def main():
    # 加载chunks数据
    all_chunk_data = load_chunks_to_chunk_data(INPUT_DIR)

    # 构建提示词
    for i in range(len(all_chunk_data)):
        if all_chunk_data[i].h2_title != "":  
            generate_prompt = (
                f"This is an image from the file '{all_chunk_data[i].source_file}', "
                f"This is the {all_chunk_data[i].position_desc}th image in the document, "
                f"located in the main section titled '{all_chunk_data[i].h1_title}'. "
                f"The sub-section is titled '{all_chunk_data[i].h2_title}', "
                f"The preceding text of this image is: {all_chunk_data[i].img_above_text},"
                f"The following text of this image is: {all_chunk_data[i].img_below_text},"
                f"The summary of this picture is:{all_chunk_data[i].img_summary}"
            )
        else:  # 一级
            generate_prompt = (
                f"This is an image from the file '{all_chunk_data[i].source_file}', "
                f"This is the {all_chunk_data[i].position_desc}th image in the document, "
                f"located in the main section titled '{all_chunk_data[i].h1_title}'. "
                f"The preceding text of this image is: {all_chunk_data[i].img_above_text},"
                f"The following text of this image is: {all_chunk_data[i].img_below_text},"
                f"The summary of this picture is:{all_chunk_data[i].img_summary}"
            )

        all_chunk_data[i].embedding_prompt = all_chunk_data[i].img_summary
        all_chunk_data[i].generate_prompt = generate_prompt

    # 创建向量索引库
    textembedding3largeconfig = TextEmbedding3LargeConfig()
    index = faiss.IndexFlatL2(textembedding3largeconfig.EMBEDDING_DIM)   # 使用 L2 距离的扁平索引（暴力搜索）
    index = faiss.IndexIDMap(index)  # 支持 add_with_ids

    # 将chunk内容转为向量，保存到向量索引库中
    def process_chunk(chunk_data):
        try:
            response = client.embeddings.create(
                input=chunk_data.embedding_prompt,
                model=textembedding3largeconfig.EMBEDDING_MODEL_NAME
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return chunk_data.chunk_id, embedding
        except Exception as e:
            error_msg = str(e)
            logger.error(f"计算向量时发生错误: {error_msg}")
            return None

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in all_chunk_data]
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is not None:
                chunk_id, embedding = result
                index.add_with_ids(embedding.reshape(1, -1), np.array([chunk_id]))
                logger.info(f"成功将chunk {i} 的内容与向量建立索引")

    # 保存chunk信息到JOSN文件
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

    # 构建ID到JSON的路径映射
    mapping_path = os.path.join(OUTPUT_DIR, "chunk_id_to_path.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(chunk_id_to_path, f, ensure_ascii=False, indent=4)

    logger.info(f"chunk_id 映射信息已保存到 {mapping_path}")
    
    # 保存向量索引库
    index_path = os.path.join(OUTPUT_DIR, "img_embedder_index.faiss")
    faiss.write_index(index, index_path)
    logger.info(f"向量索引库已保存到 {index_path}")

if __name__ == "__main__":
    main()
