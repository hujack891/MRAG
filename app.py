from flask import Flask, render_template, request, jsonify
import os
import json
from datetime import datetime
import numpy as np
import faiss
from openai import OpenAI
import sys
from config import BaseConfig, TextEmbedding3LargeConfig, MLLMConfig
from logs.log_config import setup_logging

# 初始化日志
logger = setup_logging(os.path.splitext(os.path.basename(__file__))[0])

app = Flask(__name__)

# 配置
TOP_K_RESULTS = 5  # 文本检索数量
TOP_N_RESULTS = 5  # 图片检索数量
text_weight = 0.5
image_weight = 0.5
MAX_CONTEXT_LENGTH = 4000

TEXT_DATABASE_PATH = "./index/text/v3"
IMAGE_DATABASE_PATH = "./index/image/v3"
OUTPUT_DIR = "./result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化配置
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

# 全局变量存储数据库
text_index = None
image_index = None
text_chunk_id_to_path = None
image_chunk_id_to_path = None
text_chunk_path = None
image_chunk_path = None

def load_databases():
    """加载数据库"""
    global text_index, image_index, text_chunk_id_to_path, image_chunk_id_to_path, text_chunk_path, image_chunk_path
    
    try:
        # 加载文本数据库
        if os.path.exists(TEXT_DATABASE_PATH):
            logger.info(f"正在加载文本数据库: {TEXT_DATABASE_PATH}")
            text_index_path = os.path.join(TEXT_DATABASE_PATH, "text_embedder_index.faiss")
            text_index = faiss.read_index(text_index_path)
            text_chunk_index_path = os.path.join(TEXT_DATABASE_PATH, "chunk_id_to_path.json")

            with open(text_chunk_index_path, 'r', encoding='utf-8') as f:
                text_chunk_id_to_path = json.load(f)

            text_chunk_path = TEXT_DATABASE_PATH
        else:
            logger.error("警告: 文本数据库文件不存在")
            return False
                                
        # 加载图片数据库
        if os.path.exists(IMAGE_DATABASE_PATH):
            logger.info(f"正在加载图片数据库: {IMAGE_DATABASE_PATH}")
            image_index_path = os.path.join(IMAGE_DATABASE_PATH, "img_embedder_index.faiss")
            image_index = faiss.read_index(image_index_path)
            image_chunk_index_path = os.path.join(IMAGE_DATABASE_PATH, "chunk_id_to_path.json")

            with open(image_chunk_index_path, 'r', encoding='utf-8') as f:
                image_chunk_id_to_path = json.load(f)

            image_chunk_path = IMAGE_DATABASE_PATH
        else:
            logger.error("警告: 图片数据库文件不存在")
            return False
            
        return True
            
    except Exception as e:
        logger.error(f"加载数据库时出错: {str(e)}")
        return False

def query_embedding(query):
    """生成查询向量"""
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

def format_retrieval_context(search_results, top_k_results, top_n_results):
    """格式化检索结果"""
    if not search_results:
        return "检索失败，无可用上下文信息。"
    
    context_parts = []
    context_parts.append(f"=== Retrieved Relevant Information ===")
    context_parts.append(f"Text Segments: {top_k_results}")
    context_parts.append(f"Related Images: {top_n_results}")
    context_parts.append("")
    
    for i, result in enumerate(search_results):
        if result["content_type"] == "text":
            context_parts.append(f"Text Segment {i+1}")
            context_parts.append(f"{result['promot']}")
            context_parts.append("")  
        elif result["content_type"] == "image":
            context_parts.append(f"Image Segment {i+1}")     
            context_parts.append(f"{result['embedding_prompt']}")      
            context_parts.append(f"Image URL: {result['img_url']}")
            context_parts.append("")    

    return "\n".join(context_parts)

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    """处理查询请求"""
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        top_k = int(data.get('top_k', TOP_K_RESULTS))
        top_n = int(data.get('top_n', TOP_N_RESULTS))
        
        if not query_text:
            return jsonify({'error': '请输入有效的查询问题'}), 400
        
        # 生成查询向量
        query_vector = query_embedding(query_text)
        if query_vector is None:
            return jsonify({'error': '生成查询向量失败'}), 500
        
        # 文本检索
        text_results = []
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
                
                text_results.append({
                    'rank': i + 1,
                    'content_type': 'text',
                    'distance': float(distance),
                    'similarity_score': float(similarity_score),
                    'weighted_score': float(similarity_score) * text_weight,
                    'chunk_id': chunk_data['chunk_id'],
                    'source_file': chunk_data['source_file'],
                    'content_type': chunk_data['content_type'],
                    'h1_title': chunk_data['h1_title'],
                    'h2_title': chunk_data['h2_title'],
                    'paragraph_content': chunk_data['content'],
                    'promot': chunk_data['promot'],
                })
        
        # 图片检索
        image_results = []
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
                
                image_results.append({
                    'rank': i + 1,
                    'content_type': 'image',
                    'distance': float(distance),
                    'similarity_score': float(similarity_score),
                    'weighted_score': float(similarity_score) * image_weight,
                    'chunk_id': chunk_data['chunk_id'],
                    'source_file': chunk_data['source_file'],
                    'h1_title': chunk_data['h1_title'],
                    'h2_title': chunk_data['h2_title'],
                    'h3_title': chunk_data['h3_title'],
                    'img_url': chunk_data['img_url'],
                    'alt_text': chunk_data['alt_text'],
                    'position_desc': chunk_data['position_desc'],
                    'img_above_text': chunk_data['img_above_text'],
                    'img_below_text': chunk_data['img_below_text'],
                    'img_summary': chunk_data['img_summary'],
                    'embedding_prompt': chunk_data['embedding_prompt']
                })
        
        # 合并结果并排序
        results = text_results + image_results
        results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # 格式化上下文
        context = format_retrieval_context(results, top_k, top_n)
        
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
        user_prompt = f"""Question: {query_text}
            retrieved information:
            {context}
        """
        
        # 生成回答
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
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(OUTPUT_DIR, f'results_text_and_image_{timestamp}.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        # 保存Markdown文件
        if answer:
            filename = f'answer_{timestamp}.md'
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Q: {query_text}\n")
                f.write(f"A:\n")
                f.write(f"{answer}\n")
        
        return jsonify({
            'answer': answer,
            'results': results,
            'context': context,
            'timestamp': timestamp
        })
        
    except Exception as e:
        logger.error(f"处理查询时出错: {str(e)}")
        return jsonify({'error': f'处理查询时出错: {str(e)}'}), 500

if __name__ == '__main__':
    # 加载数据库
    if load_databases():
        print("数据库加载成功，启动Web服务...")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("数据库加载失败，请检查数据库文件是否存在") 