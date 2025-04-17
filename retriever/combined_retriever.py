# retriever/combined_retriever.py
from typing import List, Dict, Optional
from .semantic_retriever import smart_semantic_retrieve as semantic_retrieve, initialize as semantic_initialize
from .keyword_retriever import keyword_retrieve_smart_fuzzy, initialize as keyword_initialize

# 全局配置
class Config:
    openai_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    json_path: Optional[str] = None
    chat_model: str = "gpt-4o"
    
# 初始化标志
_initialized: bool = False

def initialize(
    openai_api_key: str,
    pinecone_api_key: str,
    pinecone_index_name: str,
    json_path: str,
    chat_model: str = "gpt-4o"
) -> None:
    """
    初始化组合检索器
    
    Args:
        openai_api_key: OpenAI API密钥
        pinecone_api_key: Pinecone API密钥
        pinecone_index_name: Pinecone索引名称
        json_path: 段落数据的JSON文件路径
        chat_model: 使用的chat模型
    """
    global _initialized
    
    # 更新配置
    Config.openai_api_key = openai_api_key
    Config.pinecone_api_key = pinecone_api_key
    Config.pinecone_index_name = pinecone_index_name
    Config.json_path = json_path
    Config.chat_model = chat_model
    
    # 初始化语义检索器
    semantic_initialize(
        openai_api_key=openai_api_key,
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        chat_model=chat_model
    )
    
    # 初始化关键词检索器
    keyword_initialize(
        openai_api_key=openai_api_key,
        json_path=json_path,
        chat_model=chat_model
    )
    
    _initialized = True

def combined_retrieve(
    query: str,
    top_k_base: int = 20,
    top_k_high: int = 100,
    weight_semantic: float = 1.0,
    weight_keyword: float = 0.8
) -> List[Dict]:
    """
    组合检索函数，结合语义检索和关键词检索的结果
    
    Args:
        query: 查询文本
        top_k_base: 基础返回数量
        top_k_high: 高召回率模式下的返回数量
        weight_semantic: 语义检索权重
        weight_keyword: 关键词检索权重
        
    Returns:
        检索结果列表
    """
    if not _initialized:
        raise RuntimeError("请先调用initialize()进行初始化")
        
    # 动态确定top k
    def determine_top_k(query: str) -> int:
        pattern_keywords = ["多少次", "几次", "列举", "所有", "哪些", "每一回", "全部"]
        for kw in pattern_keywords:
            if kw in query:
                return top_k_high
        return top_k_base
    
    top_k = determine_top_k(query=query)
    print(f"使用 top_k={top_k}")
    
    # 获取语义检索结果
    semantic_results = semantic_retrieve(query, top_k=top_k * 2)
    for item in semantic_results:
        item["_semantic_score"] = item.get("score", 0.0)
        item["_keyword_score"] = 0.0
        item["_source"] = "semantic"
    
    # 获取关键词检索结果（使用初始化时加载的段落数据）
    keyword_results = keyword_retrieve_smart_fuzzy(query, top_k=top_k * 2)
    for item in keyword_results:
        item["_semantic_score"] = 0.0
        item["_source"] = "keyword"
    
    # 合并（按 ID 去重 + 累加得分）
    combined_dict = {}
    
    def compute_final_score(sem_score, kw_score):
        return weight_semantic * sem_score + weight_keyword * kw_score
    
    for item in semantic_results + keyword_results:
        key = item["id"]
        if key not in combined_dict:
            combined_dict[key] = item
        else:
            # 合并两个模块命中的同一段落
            combined_dict[key]["_semantic_score"] += item.get("_semantic_score", 0.0)
            combined_dict[key]["_keyword_score"] += item.get("_keyword_score", 0.0)
    
    # 计算最终得分 + 排序
    combined_list = list(combined_dict.values())
    for item in combined_list:
        item["_final_score"] = compute_final_score(
            item["_semantic_score"],
            item["_keyword_score"]
        )
    
    combined_list.sort(key=lambda x: -x["_final_score"])
    return combined_list[:top_k] 