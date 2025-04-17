# retriever/semantic_retriever.py
from openai import OpenAI
from pinecone import Pinecone
from typing import List, Dict, Optional

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
    _client = OpenAI(api_key=openai_api_key)
    _pc = Pinecone(api_key=pinecone_api_key)
    _index = _pc.Index(pinecone_index_name)

def get_query_embedding(text: str, model: Optional[str] = None) -> List[float]:
    """获取文本的embedding向量"""
    if _client is None:
        raise RuntimeError("请先调用initialize()进行初始化")
        
    response = _client.embeddings.create(
        input=[text],
        model=model or Config.embedding_model
    )
    return response.data[0].embedding

def expand_query_semantically(prompt: str, num_variants: int = 4) -> List[str]:
    """扩展查询语义"""
    if _client is None:
        raise RuntimeError("请先调用initialize()进行初始化")
        
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

def semantic_retrieve_multi_query(query_list: List[str], top_k_per_query: int = 5, final_top_k: int = 5) -> List[Dict]:
    """多查询语义检索"""
    if _index is None:
        raise RuntimeError("请先调用initialize()进行初始化")
        
    results = []
    seen_ids = set()
    
    for q in query_list:
        q_vector = get_query_embedding(q)
        response = _index.query(vector=q_vector, top_k=top_k_per_query, include_metadata=True)
        
        for match in response.matches:
            if match.id not in seen_ids:
                results.append({
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata["text"],
                    "chapter": match.metadata.get("chapter", ""),
                    "chapter_num": match.metadata.get("chapter_num", -1)
                })
                seen_ids.add(match.id)
                
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
        
    query_variants = expand_query_semantically(query)
    query_variants.insert(0, query)
    return semantic_retrieve_multi_query(query_variants, top_k_per_query=4, final_top_k=top_k)
