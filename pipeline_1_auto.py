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
from typing import Dict, Any, List, Tuple
import re
import math
from collections import Counter

# Implementation: Hybrid Search (Dense + Sparse BM25) → GPT-4o Answer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logs.log_config import setup_logging

# 初始化运行日志
logger = setup_logging(os.path.splitext(os.path.basename(__file__))[0])

# ==================== 配置区域 ====================
TOP_K_RESULTS = 5  # 文本检索数量
TOP_N_RESULTS = 5  # 图片检索数量

# Hybrid search weights
DENSE_WEIGHT = 0.6      # Dense embedding similarity weight
SPARSE_WEIGHT = 0.4     # Sparse (BM25) similarity weight

# BM25 parameters
BM25_K1 = 1.2
BM25_B = 0.75

MAX_CONTEXT_LENGTH = 4000  # Maximum context length

TEXT_DATABASE_PATH = "./index/text/withcontext"
IMAGE_DATABASE_PATH = "./index/image/nocontext"
OUTPUT_DIR = "./result/level1_auto_hybrid"
DATASETS_ORG_DIR = "datasets_org"

os.makedirs(OUTPUT_DIR, exist_ok=True)

baseconfig = BaseConfig()
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

class BM25Retriever:
    def __init__(self, k1=BM25_K1, b=BM25_B):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.text_chunk_data = []
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split by whitespace and punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
        
    def fit(self, corpus: List[str], chunk_data: List[Dict]):
        """Train BM25 model"""
        self.text_chunk_data = chunk_data
        self.corpus = [self._tokenize(doc) for doc in corpus]
        
        # Calculate document frequencies
        df = {}
        for doc in self.corpus:
            for word in set(doc):
                df[word] = df.get(word, 0) + 1
                
        # Calculate IDF
        for word, freq in df.items():
            self.idf[word] = math.log((len(self.corpus) - freq + 0.5) / (freq + 0.5))
            
        # Calculate document lengths
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        
    def search(self, query: str, top_k: int) -> List[Dict]:
        """Search using BM25"""
        query_tokens = self._tokenize(query)
        scores = []
        
        for i, doc in enumerate(self.corpus):
            score = 0
            doc_len = self.doc_len[i]
            
            for token in query_tokens:
                if token in self.idf:
                    tf = doc.count(token)
                    idf = self.idf[token]
                    score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
                    
            scores.append(score)
            
        # Get top-k results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                chunk_data = self.text_chunk_data[idx].copy()
                chunk_data.update({
                    'rank': rank + 1,
                    'content_type': 'text',
                    'bm25_score': float(scores[idx]),
                    'search_type': 'sparse'
                })
                results.append(chunk_data)
                
        return results

class HybridSearcher:
    def __init__(self):
        self.bm25_retriever = BM25Retriever()
        self.text_chunk_data = []
        self.image_chunk_data = []
        
    def build_sparse_index(self, text_chunks: List[Dict]):
        """Build BM25 sparse index for text chunks"""
        logger.info("Building sparse BM25 index...")
        
        # Prepare text corpus
        corpus = []
        for chunk in text_chunks:
            # Combine different text fields for comprehensive search
            text_content = f"{chunk.get('h1_title', '')} {chunk.get('h2_title', '')} {chunk.get('h3_title', '')} {chunk.get('content', '')} {chunk.get('promot', '')}"
            corpus.append(text_content)
            
        self.text_chunk_data = text_chunks
        
        # Build BM25 index
        self.bm25_retriever.fit(corpus, text_chunks)
        logger.info(f"Built BM25 index with {len(corpus)} documents")
        
    def sparse_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform sparse retrieval using BM25"""
        return self.bm25_retriever.search(query, top_k)
        
    def dense_search(self, query_vector: np.ndarray, text_index, image_index, 
                    text_chunk_id_to_path: Dict, image_chunk_id_to_path: Dict,
                    text_chunk_path: str, image_chunk_path: str, 
                    top_k: int, top_n: int) -> Tuple[List[Dict], List[Dict]]:
        """Perform dense retrieval using FAISS"""
        text_results = []
        image_results = []
        
        # Dense text search
        if text_index is not None:
            text_distances, text_indices = text_index.search(
                query_vector.reshape(1, -1), top_k
            )

            for i, (distance, idx) in enumerate(zip(text_distances[0], text_indices[0])):
                chunk_id = text_chunk_id_to_path[str(idx)]
                chunk_path = os.path.join(text_chunk_path, chunk_id)
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                similarity_score = 1 / (1 + distance)
            
                chunk_data.update({
                    'rank': i + 1,
                    'content_type': 'text',
                    'distance': float(distance),
                    'dense_score': float(similarity_score),
                    'search_type': 'dense'
                })
                text_results.append(chunk_data)
                
        # Dense image search  
        if image_index is not None:
            image_distances, image_indices = image_index.search(
                query_vector.reshape(1, -1), top_n
            )
            
            for i, (distance, idx) in enumerate(zip(image_distances[0], image_indices[0])):
                chunk_id = image_chunk_id_to_path[str(idx)]
                chunk_path = os.path.join(image_chunk_path, chunk_id)
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                similarity_score = 1 / (1 + distance)
            
                chunk_data.update({
                    'rank': i + 1,
                    'content_type': 'image',
                    'distance': float(distance),
                    'dense_score': float(similarity_score),
                    'search_type': 'dense'
                })
                image_results.append(chunk_data)
                
        return text_results, image_results
    
    def hybrid_fusion(self, dense_text_results: List[Dict], sparse_text_results: List[Dict], 
                     dense_image_results: List[Dict]) -> List[Dict]:
        """Fuse results from dense and sparse search strategies"""
        
        # Create a dictionary to store combined scores
        combined_results = {}
        
        # Process dense text results
        for result in dense_text_results:
            key = result.get('chunk_id', f"text_{result.get('rank', 0)}")
            if key not in combined_results:
                combined_results[key] = result.copy()
                combined_results[key]['dense_score'] = result.get('dense_score', 0)
                combined_results[key]['bm25_score'] = 0
            else:
                combined_results[key]['dense_score'] = result.get('dense_score', 0)
        
        # Process sparse text results (BM25)
        for result in sparse_text_results:
            # Try to match with dense results by content similarity
            key = None
            for existing_key in combined_results.keys():
                if (combined_results[existing_key]['content_type'] == 'text' and 
                    combined_results[existing_key].get('content', '') == result.get('content', '')):
                    key = existing_key
                    break
            
            if key:
                combined_results[key]['bm25_score'] = result.get('bm25_score', 0)
            else:
                # Add as new result if not found in dense results
                new_key = f"sparse_{result.get('rank', 0)}"
                combined_results[new_key] = result.copy()
                combined_results[new_key]['dense_score'] = 0
                combined_results[new_key]['bm25_score'] = result.get('bm25_score', 0)
        
        # Process dense image results
        for result in dense_image_results:
            key = result.get('chunk_id', f"image_{result.get('rank', 0)}")
            if key not in combined_results:
                combined_results[key] = result.copy()
                combined_results[key]['dense_score'] = result.get('dense_score', 0)
                combined_results[key]['bm25_score'] = 0
            else:
                combined_results[key]['dense_score'] = result.get('dense_score', 0)
        
        # Calculate hybrid scores
        final_results = []
        for key, result in combined_results.items():
            # Normalize scores to [0, 1] range
            dense_norm = min(result.get('dense_score', 0), 1.0)
            # Normalize BM25 scores (typically range from 0 to ~10+)
            bm25_norm = min(result.get('bm25_score', 0) / 10.0, 1.0)
            
            # Calculate hybrid score
            hybrid_score = (
                DENSE_WEIGHT * dense_norm + 
                SPARSE_WEIGHT * bm25_norm
            )
            
            result['hybrid_score'] = hybrid_score
            result['score_breakdown'] = {
                'dense': dense_norm,
                'bm25': bm25_norm
            }
            
            result['final_score'] = hybrid_score
                
            final_results.append(result)
        
        # Sort by final score
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        return final_results

# Initialize hybrid searcher
hybrid_searcher = HybridSearcher()

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
    
def format_retrieval_context(search_results: list, TOP_K_RESULTS: int, TOP_N_RESULTS: int) -> str:
    """格式化检索内容"""
    if not search_results:
        return "检索失败，无可用上下文信息"
    
    context_parts = []
    
    context_parts.append(f"=== Retrieved Relevant Information (Hybrid Search) ===")
    context_parts.append(f"Text Segments: {TOP_K_RESULTS}")
    context_parts.append(f"Related Images: {TOP_N_RESULTS}")
    context_parts.append(f"Search Strategy: Dense + Sparse BM25 Fusion")
    context_parts.append("")
    
    for i, result in enumerate(search_results):
        if result["content_type"] == "text":
            context_parts.append(f"Text Segment {i+1} (Score: {result.get('final_score', 0):.3f})")
            context_parts.append(f"Search Methods: Dense={result.get('score_breakdown', {}).get('dense', 0):.3f}, "
                                f"BM25={result.get('score_breakdown', {}).get('bm25', 0):.3f}")
            context_parts.append(f"{result.get('promot', result.get('content', ''))}")
            context_parts.append("")  
            
        elif result["content_type"] == "image":
            context_parts.append(f"Image Segment {i+1} (Score: {result.get('final_score', 0):.3f})")   
            context_parts.append(f"Search Methods: Dense={result.get('score_breakdown', {}).get('dense', 0):.3f}")
            context_parts.append(f"{result.get('generate_prompt', '')}")      
            context_parts.append(f"Image URL: {result.get('img_url', '')}")
            context_parts.append("")    

    return "\n".join(context_parts)

def process_single_question(query: str, text_index, image_index, text_chunk_id_to_path, image_chunk_id_to_path, text_chunk_path, image_chunk_path, top_k: int, top_n: int) -> Dict[str, Any]:
    """处理单个问题并返回结果 - 使用混合搜索"""
    logger.info(f"开始处理问题 (混合搜索): {query}")
    
    # 将问题转换为向量
    query_vector = query_embedding(query)
    if query_vector is None:
        return {"error": "向量生成失败"}
    
    # 1. Dense search (original FAISS search)
    logger.info("执行密集向量搜索...")
    dense_text_results, dense_image_results = hybrid_searcher.dense_search(
        query_vector, text_index, image_index, text_chunk_id_to_path, 
        image_chunk_id_to_path, text_chunk_path, image_chunk_path, top_k, top_n
    )
    
    # 2. Sparse search (BM25 based)
    logger.info("执行BM25稀疏搜索...")
    sparse_text_results = hybrid_searcher.sparse_search(query, top_k)
    
    # 3. Hybrid fusion
    logger.info("融合密集和稀疏搜索结果...")
    results = hybrid_searcher.hybrid_fusion(
        dense_text_results, sparse_text_results, 
        dense_image_results
    )
    
    # Limit to desired number of results
    total_results = min(len(results), top_k + top_n)
    results = results[:total_results]

    # 保存检索结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(OUTPUT_DIR, f'hybrid_results_{timestamp}.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 生成上下文
    context = format_retrieval_context(results, top_k, top_n)

    # 构建提示词
    system_prompt = """Based on the retrieved text segments, image content, and their associated metadata (obtained through hybrid search combining dense embeddings and sparse BM25 keyword matching) relevant to the user's question, generate content that conforms to the following Markdown format specification, while preserving the global logical sequence and contextual relationships.

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
        5. The retrieved information is ranked by hybrid relevance score (combining multiple search strategies)
        6. Consider the search method breakdown when integrating information
    """

    user_prompt = f"""Question: {query}
        Retrieved information (Hybrid Search):
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
            filename = f'{safe_query}_hybrid_answer_{timestamp}.md'
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Q: {query}\n")
                f.write(f"A (Hybrid Search):\n")
                f.write(f"{answer}\n")
            logger.info(f"答案已保存到: {filepath}")
            
            return {
                "question": query,
                "answer": answer,
                "results_file": results_file,
                "answer_file": filepath,
                "retrieval_results": results,
                "search_type": "hybrid"
            }
        else:
            logger.warning("生成的答案为空")
            return {"error": "生成的答案为空"}
        
    except Exception as e:
        error_msg = f"生成答案时出错: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

def load_all_chunks_for_sparse_index(text_chunk_path: str, text_chunk_id_to_path: Dict):
    """Load all text chunks for building BM25 sparse index"""
    logger.info("Loading text chunks for BM25 indexing...")
    
    # Load text chunks
    text_chunks = []
    for chunk_id in text_chunk_id_to_path.values():
        chunk_path = os.path.join(text_chunk_path, chunk_id)
        try:
            with open(chunk_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                text_chunks.append(chunk_data)
        except Exception as e:
            logger.error(f"Error loading text chunk {chunk_id}: {str(e)}")
    
    logger.info(f"Loaded {len(text_chunks)} text chunks")
    
    # Build BM25 sparse index for text chunks
    hybrid_searcher.build_sparse_index(text_chunks)

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
    
    # Load all text chunks for BM25 indexing
    load_all_chunks_for_sparse_index(text_chunk_path, text_chunk_id_to_path)
    
    # 读取所有问题
    logger.info(f"正在从 {DATASETS_ORG_DIR} 读取问题...")
    questions = load_questions_from_datasets(DATASETS_ORG_DIR)
    
    if not questions:
        logger.error("未找到任何问题，程序退出")
        return
    
    # 问答循环
    while True:
        print("\n=== 混合搜索批量处理模式 ===")
        print("1. 处理所有问题")
        print("2. 随机处理指定数量的问题")
        print("3. 手动输入问题")
        print("4. 调整搜索权重")
        print("5. 退出")
        
        choice = input("请选择模式 (1-5): ").strip()
        
        if choice == '5':
            logger.info("感谢使用！")
            break
        elif choice == '4':
            # 调整搜索权重
            print(f"\n当前权重设置:")
            print(f"Dense Weight: {DENSE_WEIGHT}")
            print(f"Sparse BM25 Weight: {SPARSE_WEIGHT}")
            
            try:
                new_dense = float(input(f"新的Dense权重 (当前: {DENSE_WEIGHT}): ") or DENSE_WEIGHT)
                new_sparse = float(input(f"新的Sparse BM25权重 (当前: {SPARSE_WEIGHT}): ") or SPARSE_WEIGHT)
                
                # Normalize weights
                total_weight = new_dense + new_sparse
                if total_weight > 0:
                    DENSE_WEIGHT = new_dense / total_weight
                    SPARSE_WEIGHT = new_sparse / total_weight
                    
                    logger.info(f"权重已更新: Dense={DENSE_WEIGHT:.3f}, Sparse BM25={SPARSE_WEIGHT:.3f}")
                else:
                    print("权重总和必须大于0")
            except ValueError:
                print("无效的权重值")
            continue
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


