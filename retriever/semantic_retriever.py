# retriever/semantic_retriever.py
import os
import logging
from openai import OpenAI
import time
from typing import List, Dict, Optional

# 配置日志
logger = logging.getLogger("xiyouji-semantic-retriever")

# 尝试导入Pinecone，并处理可能的导入错误
try:
    from pinecone import Pinecone
    logger.info("成功导入pinecone库")
except ImportError as e:
    logger.error(f"导入pinecone库失败: {str(e)}")
    logger.error("请确保已安装pinecone库 (不是pinecone-client)")
    raise

# 全局配置
class Config:
    openai_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: str = "xiyouji-embedding"
    embedding_model: str = "text-embedding-3-large"
    chat_model: str = "gpt-4o"

# 全局客户端实例
_client: Optional[OpenAI] = None
_pc: Optional[Pinecone] = None
_index = None

def initialize(
    openai_api_key: str,
    pinecone_api_key: str,
    pinecone_index_name: str,
    embedding_model: str = "text-embedding-3-large",
    chat_model: str = "gpt-4o"
) -> None:
    """
    初始化检索器配置
    
    Args:
        openai_api_key: OpenAI API密钥
        pinecone_api_key: Pinecone API密钥
        pinecone_index_name: Pinecone索引名称
        embedding_model: 使用的embedding模型
        chat_model: 使用的chat模型
    """
    global _client, _pc, _index
    
    # 更新配置
    Config.openai_api_key = openai_api_key
    Config.pinecone_api_key = pinecone_api_key
    Config.pinecone_index_name = pinecone_index_name
    Config.embedding_model = embedding_model
    Config.chat_model = chat_model
    
    # 初始化客户端
    try:
        logger.info("正在初始化OpenAI客户端...")
        _client = OpenAI(api_key=openai_api_key)
        
        logger.info("正在初始化Pinecone客户端...")
        _pc = Pinecone(api_key=pinecone_api_key)
        
        logger.info(f"正在连接Pinecone索引: {pinecone_index_name}")
        _index = _pc.Index(pinecone_index_name)
        logger.info("初始化完成!")
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}", exc_info=True)
        raise

def get_query_embedding(text: str, model: Optional[str] = None) -> List[float]:
    """获取文本的embedding向量"""
    if _client is None:
        raise RuntimeError("请先调用initialize()进行初始化")
    
    # 添加重试逻辑
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = _client.embeddings.create(
                input=[text],
                model=model or Config.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"获取embedding失败，尝试重试 ({attempt+1}/{max_retries}): {str(e)}")
                time.sleep(1 * (attempt + 1))  # 指数级退避
            else:
                logger.error(f"获取embedding失败: {str(e)}", exc_info=True)
                raise

def expand_query_semantically(prompt: str, num_variants: int = 4) -> List[str]:
    """扩展查询语义"""
    if _client is None:
        raise RuntimeError("请先调用initialize()进行初始化")
    
    # 添加重试逻辑    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            system_prompt = (
                "你是一个智能搜索助手。请根据用户提出的问题，生成一些语义不同但含义相近的表达方式，"
                f"用于增强语义搜索召回。每个表达不要太长，直接列出 {num_variants} 个不同表达。"
            )
            
            response = _client.chat.completions.create(
                model=Config.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"原始问题：{prompt}\n请列出等价或近似问题："}
                ],
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            return [line.strip(" 1234567890.-、：") for line in content.strip().split("\n") if line.strip()][:num_variants]
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"扩展查询失败，尝试重试 ({attempt+1}/{max_retries}): {str(e)}")
                time.sleep(1 * (attempt + 1))
            else:
                logger.error(f"扩展查询失败: {str(e)}", exc_info=True)
                # 如果无法扩展，返回原始查询
                return [prompt]

def semantic_retrieve_multi_query(query_list: List[str], top_k_per_query: int = 5, final_top_k: int = 5) -> List[Dict]:
    """多查询语义检索"""
    if _index is None:
        raise RuntimeError("请先调用initialize()进行初始化")
    
    results = []
    seen_ids = set()
    
    try:
        for q in query_list:
            try:
                q_vector = get_query_embedding(q)
                response = _index.query(vector=q_vector, top_k=top_k_per_query, include_metadata=True)
                
                for match in response.matches:
                    if match.id not in seen_ids:
                        results.append({
                            "id": match.id,
                            "score": match.score,
                            "text": match.metadata.get("text", ""),
                            "chapter": match.metadata.get("chapter", ""),
                            "chapter_num": match.metadata.get("chapter_num", -1)
                        })
                        seen_ids.add(match.id)
            except Exception as e:
                logger.error(f"处理查询 '{q}' 时出错: {str(e)}")
                continue
    except Exception as e:
        logger.error(f"多查询检索失败: {str(e)}", exc_info=True)
        return []
                
    return sorted(results, key=lambda x: -x["score"])[:final_top_k]

def smart_semantic_retrieve(query: str, paragraphs: Optional[List[Dict]] = None, top_k: int = 5) -> List[Dict]:
    """
    智能语义检索
    
    Args:
        query: 查询文本
        paragraphs: 段落列表，在语义检索中不使用但保留参数以保持接口一致性
        top_k: 返回结果数量
        
    Returns:
        检索结果列表
    """
    if _client is None or _index is None:
        raise RuntimeError("请先调用initialize()进行初始化")
    
    try:    
        query_variants = expand_query_semantically(query)
        # 确保原始查询在列表中
        if query not in query_variants:
            query_variants.insert(0, query)
        return semantic_retrieve_multi_query(query_variants, top_k_per_query=4, final_top_k=top_k)
    except Exception as e:
        logger.error(f"智能语义检索失败: {str(e)}", exc_info=True)
        # 如果失败，尝试直接用原始查询进行检索
        try:
            return semantic_retrieve_multi_query([query], top_k_per_query=top_k, final_top_k=top_k)
        except:
            logger.error("备用检索也失败", exc_info=True)
            return []
