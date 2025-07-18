import numpy as np
import faiss
import json
import os
from datetime import datetime
from openai import OpenAI
import sys
from config import BaseConfig, TextEmbedding3LargeConfig, MLLMConfig
from typing import Dict, Any

# Implementation: Question → Embedding Vector → Vector Database Retrieval → GPT-4o Answer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logs.log_config import setup_logging

# 初始化运行日志
logger = setup_logging(os.path.splitext(os.path.basename(__file__))[0])

# ==================== 配置区域 ====================
TOP_K_RESULTS = 5  # 文本检索数量
TOP_N_RESULTS = 5  # 图片检索数量

text_weight = 0.5  # 弥补文本和图片的差距
image_weight = 0.5

MAX_CONTEXT_LENGTH = 4000  # Maximum context length

TEXT_DATABASE_PATH = "./index/text/v3"
IMAGE_DATABASE_PATH = "./index/image/v3"
OUTPUT_DIR = "./result/level3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

baseconfig= BaseConfig()
textembedding3largeconfig = TextEmbedding3LargeConfig()
mllmconfig = MLLMConfig()

client_embedding = OpenAI(
    base_url=baseconfig.EMBEDDING_URL,
    api_key=baseconfig.EMBEDDING_API
)

client_gpt4o = OpenAI(
    base_url=baseconfig.MLLM_URL,
    api_key=baseconfig.MLLM_API
)

def query_embedding(query):
    try:
        response = client_embedding.embeddings.create(
            input=query,
            model=textembedding3largeconfig.EMBEDDING_MODEL_NAME
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return embedding
    except Exception as e:
        error_msg = str(e)
        logger.error(f"生成查询向量时发生错误: {error_msg}")
        return None
    

def format_retrieval_context(search_results:list,TOP_K_RESULTS:int,TOP_N_RESULTS:int) -> str:
    """
    将检索结果格式化为上下文信息，供后续AI处理
    
    Args:
        search_results: 混合搜索结果
        
    Returns:
        格式化的上下文字符串
    """
    if not search_results:
        return "检索失败，无可用上下文信息。"
    
    context_parts = []
    

    context_parts.append(f"=== Retrieved Relevant Information ===")
    context_parts.append(f"Text Segments: {TOP_K_RESULTS}")
    context_parts.append(f"Related Images: {TOP_N_RESULTS}")
    context_parts.append("")
    
    for i, result in enumerate(search_results):
        if result["content_type"] == "text":
            context_parts.append(f"Text Segment {i+1}")
            context_parts.append(f"{result['promot']}")
            context_parts.append("")  

            
        elif result["content_type"] == "image":
            context_parts.append(f"Image Segment {i+1}")     
            context_parts.append(f"{result['generate_prompt']}")      
            context_parts.append(f"Image URL: {result['img_url']}")
            context_parts.append("")    

    return "\n".join(context_parts)



def main():
    # 读取数据库
    try:
        # 加载文本数据库
        if os.path.exists(TEXT_DATABASE_PATH):
            logger.info(f"正在加载文本数据库: {TEXT_DATABASE_PATH}")
            text_index = os.path.join(TEXT_DATABASE_PATH, "text_embedder_index.faiss")
            text_index = faiss.read_index(text_index)
            text_chunk_index_path = os.path.join(TEXT_DATABASE_PATH, "chunk_id_to_path.json")

            with open(text_chunk_index_path, 'r', encoding='utf-8') as f:
                text_chunk_id_to_path = json.load(f)

            text_chunk_path = TEXT_DATABASE_PATH
        else:
            logger.error("警告: 文本数据库文件不存在")
            return
                                
        # 加载图片数据库
        if os.path.exists(IMAGE_DATABASE_PATH):
            logger.info(f"正在加载图片数据库: {IMAGE_DATABASE_PATH}")
            image_index = os.path.join(IMAGE_DATABASE_PATH, "img_embedder_index.faiss")
            image_index = faiss.read_index(image_index)
            image_chunk_index_path = os.path.join(IMAGE_DATABASE_PATH, "chunk_id_to_path.json")

            with open(image_chunk_index_path, 'r', encoding='utf-8') as f:
                image_chunk_id_to_path = json.load(f)

            image_chunk_path = IMAGE_DATABASE_PATH
        else:
            logger.error("警告: 图片数据库文件不存在")
            return
            
    except Exception as e:

        logger.error(f"加载数据库时出错: {str(e)}")
        return
    while True:

        # 输入问题
        query = input("请输入您的查询问题 (输入 'quit' 退出): ").strip()
        if query.lower() in ['quit', 'exit', '退出', 'q']:
            print("感谢使用！")
            break
        if not query:
            print("请输入有效的查询问题")
            continue    

        # 输入文本检索数量
        top_k_input = input(f"请输入文本检索数量 (默认为{TOP_K_RESULTS}): ").strip()
        try:
            top_k = int(top_k_input) if top_k_input else TOP_K_RESULTS
            if top_k <= 0:
                top_k = TOP_K_RESULTS

        except ValueError:
            top_k = TOP_K_RESULTS

        # 输入图片检索数量
        top_n_input = input(f"请输入图片检索数量 (默认为{TOP_N_RESULTS}): ").strip()
        try:
            top_n = int(top_n_input) if top_n_input else TOP_N_RESULTS
            if top_n <= 0:
                top_n = TOP_N_RESULTS
        except ValueError:
            top_n = TOP_N_RESULTS


        # 将问题转换为向量
        query_vector = query_embedding(query)
        
        # 文本向量数据库检索
        text_results = []
        image_results = []
        if text_index is not None:
            print(f"正在搜索文本数据库... (top_k={top_k})")
            text_distances, text_indices = text_index.search(
                query_vector.reshape(1, -1), top_k
            )

            
            for i, (distance, idx) in enumerate(zip(text_distances[0], text_indices[0])):
                # 通过idx获取chunk_id
                chunk_id = text_chunk_id_to_path[str(idx)]
                # 通过chunk_id获取chunk_path
                chunk_path = os.path.join(text_chunk_path, chunk_id)
                # 读取chunk_path
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                # 将chunk_data中的text_content作为metadata
                similarity_score = 1 / (1 + distance)  # 转换为相似度分数
            
                text_results.append({
                    'rank': i + 1,
                    'content_type': 'text',
                    'distance': float(distance),
                    'similarity_score': float(similarity_score),
                    "weighted_score": float(similarity_score) * text_weight,
                    "chunk_id": chunk_data['chunk_id'],
                    "source_file": chunk_data['source_file'],
                    "content_type": chunk_data['content_type'],
                    "h1_title": chunk_data['h1_title'],         # 可留空或使用默认标识
                    "h2_title": chunk_data['h2_title'],
                    "h1_content": chunk_data['h1_content'],
                    "h2_content": chunk_data['h2_content'],
                    "promot": chunk_data['promot'],                  
                })  
        if image_index is not None:
            print(f"正在搜索图片数据库... (top_n={top_n})")
            image_distances, image_indices = image_index.search(
                query_vector.reshape(1, -1), top_n
            )
            
            for i, (distance, idx) in enumerate(zip(image_distances[0], image_indices[0])):
                # 通过idx获取chunk_id
                print(idx)
                chunk_id = image_chunk_id_to_path[str(idx)]
                # 通过chunk_id获取chunk_path
                chunk_path = os.path.join(image_chunk_path, chunk_id)
                # 读取chunk_path
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                # 将chunk_data中的text_content作为metadata
                similarity_score = 1 / (1 + distance)  # 转换为相似度分数
            
                image_results.append({
                    'rank': i + 1,
                    'content_type': 'image',
                    'distance': float(distance),
                    'similarity_score': float(similarity_score),
                    "weighted_score": float(similarity_score) * image_weight,
                    "chunk_id": chunk_data['chunk_id'],
                    "source_file": chunk_data['source_file'],
                    "h1_title": chunk_data['h1_title'],         # 可留空或使用默认标识
                    "h2_title": chunk_data['h2_title'],
                    "h3_title": chunk_data['h3_title'],
                    "img_url": chunk_data['img_url'],
                    "alt_text": chunk_data['alt_text'],
                    "position_desc": chunk_data['position_desc'],
                    "img_above_text": chunk_data['img_above_text'],
                    "img_below_text": chunk_data['img_below_text'],
                    "img_summary": chunk_data['img_summary'],   
                    "embedding_prompt": chunk_data['embedding_prompt'],     
                    "generate_prompt": chunk_data['generate_prompt']     
                })             


        # 将检索结果记录

        # 将文本检索结果和图片检索结果合并
        results = text_results + image_results

        # 将结果按相似度分数从高到低排序
        results.sort(key=lambda x: x['weighted_score'], reverse=True)
        # 将结果记录到results.json文件中
        with open(os.path.join(OUTPUT_DIR, f'results_text_and_image_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        # 将检索结果拼接成上下文
        context = format_retrieval_context(results,TOP_K_RESULTS,TOP_N_RESULTS)

        # 构建系统提示词
        system_prompt = """Based on the retrieved text segments, image content, and their associated metadata (such as source document sections and image URLs) relevant to the user's question, generate content that conforms to the following Markdown format specification, while preserving the global logical sequence and contextual relationships.

            The format is as follows:
            Text Paragraph 1  
            ![Image 1 Description](Image 1 URL)  
            Text Paragraph 2  
            ![Image 2 Description](Image 2 URL)  
            ...

            Requirements:
            1. Integrate text and image information naturally, ensuring logical flow
            2. Use retrieved image URLs exactly as provided
            3. Create meaningful image descriptions based on AI summaries
            4. Maintain coherent narrative structure
            5. Preserve source context and relationships
            6. Answer the user's question comprehensively using all relevant information
            7.The retrieved information is sorted by similarity (descending). Prioritize the most relevant items at the top.
        """

        # 构建用户提示词
        user_prompt = f"""Question: {query}
            retrieved information:
            {context}
        """
        
        try:
            response = client_gpt4o.chat.completions.create(
                model=mllmconfig.MMLLM,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                    ]
                        
                    }
                ],
                max_tokens=5000,
                temperature=0.5
            )
            
            answer = response.choices[0].message.content
            print(f"Answer generation completed, length: {len(answer)} characters")
            
            # 确保生成的答案被保存为Markdown文件
            if answer:
                filename = f'answer_{datetime.now().strftime("%Y%m_%H%M%S")}.md'
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Q: {query}\n")
                    f.write(f"A:\n")
                    f.write(f"{answer}\n")
                print(f"答案已保存到: {filepath}")
            else:
                print("生成的答案为空,未保存为Markdown文件。")
            
            return answer
            
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            raise
 


if __name__ == "__main__":
    main()


