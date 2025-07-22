import numpy as np
import faiss
import json
import os
import random
import glob
from datetime import datetime
from openai import OpenAI
import sys
from config import BaseConfig, TextEmbedding3LargeConfig, MLLMConfig
from typing import Dict, Any, List

# Implementation: Question → Embedding Vector → Vector Database Retrieval → GPT-4o Answer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logs.log_config import setup_logging

# 初始化运行日志
logger = setup_logging(os.path.splitext(os.path.basename(__file__))[0])

# ==================== 配置区域 ====================
TOP_K_RESULTS = 5  # 文本检索数量
TOP_N_RESULTS = 5  # 图片检索数量

text_weight = 0.5  
image_weight = 0.5

MAX_CONTEXT_LENGTH = 4000  # Maximum context length

TEXT_DATABASE_PATH = "./index/text/withcontext"
IMAGE_DATABASE_PATH = "./index/image/withcontext"
OUTPUT_DIR = "./result/level3_auto_hybrido"
DATASETS_ORG_DIR = "datasets_org"

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

def load_questions_from_datasets(datasets_dir: str) -> List[Dict[str, str]]:
    """从datasets_org文件夹中读取所有md文件的问题"""
    questions = []
    datasets_md_files = glob.glob(os.path.join(datasets_dir, "*.md"))
    
    for md_file in datasets_md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                # 检查第一行是否以Q:开头
                if first_line.startswith('Q: '):
                    question = first_line[3:].strip()  # 去掉"Q:"前缀
                    questions.append({
                        'file': os.path.basename(md_file),
                        'question': question
                    })
                    logger.info(f"从 {os.path.basename(md_file)} 提取问题: {question}")
        except Exception as e:
            logger.error(f"读取文件 {md_file} 时出错: {str(e)}")
    
    logger.info(f"总共从 {len(datasets_md_files)} 个文件中提取了 {len(questions)} 个问题")
    return questions

def query_embedding(query):
    """将问题转为向量"""
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
    """格式化检索内容"""
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

def process_single_question(query: str, text_index, image_index, text_chunk_id_to_path, image_chunk_id_to_path, text_chunk_path, image_chunk_path, top_k: int, top_n: int) -> Dict[str, Any]:
    """处理单个问题并返回结果"""
    logger.info(f"开始处理问题: {query}")
    
    # 将问题转换为向量
    query_vector = query_embedding(query)
    if query_vector is None:
        return {"error": "向量生成失败"}
    
    # 文本向量数据库检索
    text_results = []
    image_results = []
    
    if text_index is not None:
        logger.info(f"正在搜索文本数据库... (top_k={top_k})")
        text_distances, text_indices = text_index.search(
            query_vector.reshape(1, -1), top_k
        )

        for i, (distance, idx) in enumerate(zip(text_distances[0], text_indices[0])):
            chunk_id = text_chunk_id_to_path[str(idx)]
            chunk_path = os.path.join(text_chunk_path, chunk_id)
            with open(chunk_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            similarity_score = 1 / (1 + distance)
        
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
        logger.info(f"正在搜索图片数据库... (top_n={top_n})")
        image_distances, image_indices = image_index.search(
            query_vector.reshape(1, -1), top_n
        )
        
        for i, (distance, idx) in enumerate(zip(image_distances[0], image_indices[0])):
            chunk_id = image_chunk_id_to_path[str(idx)]
            chunk_path = os.path.join(image_chunk_path, chunk_id)
            with open(chunk_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            similarity_score = 1 / (1 + distance)
        
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

    # 合并结果并排序
    results = text_results + image_results
    results.sort(key=lambda x: x['weighted_score'], reverse=True)

    # 保存检索结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(OUTPUT_DIR, f'results_{timestamp}.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 生成上下文
    context = format_retrieval_context(results, top_k, top_n)

    # 构建提示词
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
        3. Maintain coherent narrative structure
        4. Preserve source context and relationships
        5. The retrieved information is sorted by similarity (descending). 
    """

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
        logger.info(f"答案生成完成，长度: {len(answer)} 字符")
        
        # 保存答案
        if answer:
            safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
            filename = f'{safe_query}_answer_{timestamp}.md'
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Q: {query}\n")
                f.write(f"A:\n")
                f.write(f"{answer}\n")
            logger.info(f"答案已保存到: {filepath}")
            
            return {
                "question": query,
                "answer": answer,
                "results_file": results_file,
                "answer_file": filepath,
                "retrieval_results": results
            }
        else:
            logger.warning("生成的答案为空")
            return {"error": "生成的答案为空"}
        
    except Exception as e:
        error_msg = f"生成答案时出错: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

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
    
    # 读取所有问题
    logger.info(f"正在从 {DATASETS_ORG_DIR} 读取问题...")
    questions = load_questions_from_datasets(DATASETS_ORG_DIR)
    
    if not questions:
        logger.error("未找到任何问题，程序退出")
        return
    
    # 问答循环
    while True:
        print("\n=== 自动批量处理模式 ===")
        print("1. 处理所有问题")
        print("2. 随机处理指定数量的问题")
        print("3. 手动输入问题")
        print("4. 退出")
        
        choice = input("请选择模式 (1-4): ").strip()
        
        if choice == '4':
            logger.info("感谢使用！")
            break
        elif choice == '3':
            # 手动输入问题模式
            query = input("请输入您的查询问题: ").strip()
            if not query:
                print("请输入有效的查询问题")
                continue
            questions_to_process = [{'file': 'manual_input', 'question': query}]
        elif choice == '1':
            # 处理所有问题
            questions_to_process = questions
        elif choice == '2':
            # 随机处理指定数量
            try:
                num_questions = int(input(f"请输入要随机处理的问题数量 (1-{len(questions)}): ").strip())
                if num_questions <= 0 or num_questions > len(questions):
                    print(f"请输入1到{len(questions)}之间的数字")
                    continue
                questions_to_process = random.sample(questions, num_questions)
                logger.info(f"随机选择了 {num_questions} 个问题进行处理")
            except ValueError:
                print("请输入有效的数字")
                continue
        else:
            print("无效选择，请重新输入")
            continue
        
        # 获取检索参数
        top_k_input = input(f"请输入文本检索数量 (默认为{TOP_K_RESULTS}): ").strip()
        try:
            top_k = int(top_k_input) if top_k_input else TOP_K_RESULTS
            if top_k <= 0:
                top_k = TOP_K_RESULTS
        except ValueError:
            top_k = TOP_K_RESULTS

        top_n_input = input(f"请输入图片检索数量 (默认为{TOP_N_RESULTS}): ").strip()
        try:
            top_n = int(top_n_input) if top_n_input else TOP_N_RESULTS
            if top_n <= 0:
                top_n = TOP_N_RESULTS
        except ValueError:
            top_n = TOP_N_RESULTS
        
        # 批量处理问题
        logger.info(f"\n开始处理 {len(questions_to_process)} 个问题...")
        
        all_results = []
        successful_count = 0
        failed_count = 0
        
        for i, question_data in enumerate(questions_to_process, 1):
            question = question_data['question']
            source_file = question_data['file']
            
            logger.info(f"\n[{i}/{len(questions_to_process)}] 处理问题: {question}")
            logger.info(f"来源文件: {source_file}")
            
            try:
                result = process_single_question(
                    query=question,
                    text_index=text_index,
                    image_index=image_index,
                    text_chunk_id_to_path=text_chunk_id_to_path,
                    image_chunk_id_to_path=image_chunk_id_to_path,
                    text_chunk_path=text_chunk_path,
                    image_chunk_path=image_chunk_path,
                    top_k=top_k,
                    top_n=top_n
                )
                
                if "error" in result:
                    logger.info(f"处理失败: {result['error']}")
                    failed_count += 1
                else:
                    logger.info(f"处理成功")
                    logger.info(f"   答案文件: {result['answer_file']}")
                    logger.info(f"   检索结果文件: {result['results_file']}")
                    logger.info(f"   答案长度: {len(result['answer'])} 字符")

                    # 输出答案和JSON结果
                    logger.info(f"\n=== 问题 ===\n{question}")
                    logger.info(f"\n=== 答案 ===\n{result['answer']}")
                    logger.info(f"\n=== 检索结果JSON ===\n{json.dumps(result['retrieval_results'][:3], ensure_ascii=False, indent=2)}...")
                    
                    successful_count += 1
                
                result['source_file'] = source_file
                result['processing_order'] = i
                all_results.append(result)
                
            except Exception as e:
                error_msg = f"处理问题时发生异常: {str(e)}"
                logger.error(f"❌ {error_msg}")
                logger.error(error_msg)
                failed_count += 1
                all_results.append({
                    'source_file': source_file,
                    'question': question,
                    'processing_order': i,
                    'error': error_msg
                })
        
        logger.info(f"\n=== 批量处理完成 ===")
        logger.info(f"总问题数: {len(questions_to_process)}")
        logger.info(f"成功处理: {successful_count}")
        logger.info(f"处理失败: {failed_count}")
        logger.info(f"成功率: {successful_count/len(questions_to_process)*100:.1f}%")
        logger.info("-" * 50)
 
if __name__ == "__main__":
    main()


