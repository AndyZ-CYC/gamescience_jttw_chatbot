# retriever/keyword_retriever.py
import json
import re
from typing import List, Dict, Optional
from rapidfuzz import fuzz
from openai import OpenAI

# 全局配置
class Config:
    openai_api_key: Optional[str] = None
    chat_model: str = "gpt-4o"
    json_path: Optional[str] = None

# 全局客户端实例
_client: Optional[OpenAI] = None
_paragraphs: Optional[List[Dict]] = None

def load_paragraphs(json_path: str) -> List[Dict]:
    """加载段落数据"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def initialize(
    openai_api_key: str,
    json_path: str,
    chat_model: str = "gpt-4o"
) -> None:
    """
    初始化关键词检索器
    
    Args:
        openai_api_key: OpenAI API密钥
        json_path: 段落数据的JSON文件路径
        chat_model: 使用的chat模型
    """
    global _client, _paragraphs
    
    # 更新配置
    Config.openai_api_key = openai_api_key
    Config.chat_model = chat_model
    Config.json_path = json_path
    
    # 初始化客户端
    _client = OpenAI(api_key=openai_api_key)
    _paragraphs = load_paragraphs(json_path)

def extract_keywords_gpt(question: str, max_keywords: int = 5) -> List[str]:
    """使用GPT提取关键词"""
    if _client is None:
        raise RuntimeError("请先调用initialize()进行初始化")
        
    system_prompt = (
        "你是一个关键词提取工具，请根据用户提出的问题提取关键词。关键词可以是人名、地点、动词、名词等，"
        f"用于帮助文本匹配。请只返回不超过 {max_keywords} 个关键词，直接用逗号分隔返回即可。"
        f"请注意，提取关键词的时候尽量关注问题本身，不要对一些说明性语句（如，请列出原文，请简述前因后果，等）提取关键词"
    )

    user_prompt = f"问题：{question}\n提取关键词："

    response = _client.chat.completions.create(
        model=Config.chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    content = response.choices[0].message.content.strip()
    keywords = [kw.strip() for kw in content.split(",") if kw.strip()]
    return keywords[:max_keywords]

def expand_keywords_gpt(keywords: List[str], max_variants_per_word: int = 3) -> Dict[str, List[str]]:
    """对关键词进行同义扩展"""
    if _client is None:
        raise RuntimeError("请先调用initialize()进行初始化")
        
    system_prompt = (
        f"你是一个中文同义词扩展工具。请为给定的关键词生成不超过 {max_variants_per_word} 个等价或近义表达，"
        "包括同义词、简称、俗称、常用错写等。格式为：关键词: 近义词1, 近义词2, 近义词3"
    )

    user_prompt = "关键词列表：\n" + "\n".join(keywords)

    response = _client.chat.completions.create(
        model=Config.chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    content = response.choices[0].message.content.strip().split("\n")
    variant_map = {}
    for line in content:
        if ":" in line:
            key, variants = line.split(":", 1)
            variant_map[key.strip()] = [v.strip() for v in variants.split(",") if v.strip()]
    return variant_map

def keyword_match_fuzzy(all_keywords: List[str], paragraphs: Optional[List[Dict]] = None, top_k: int = 10) -> List[Dict]:
    """模糊匹配关键词"""
    if _paragraphs is None and paragraphs is None:
        raise RuntimeError("请先调用initialize()进行初始化或提供paragraphs参数")
        
    # 优先使用传入的paragraphs，如果没有则使用全局的_paragraphs
    target_paragraphs = paragraphs if paragraphs is not None else _paragraphs
    
    matches = []

    for para in target_paragraphs:
        scores = [fuzz.partial_ratio(kw, para["text"]) / 100 for kw in all_keywords]
        avg_score = sum(scores) / len(scores)
        if avg_score > 0:
            matches.append({
                **para,
                "_keyword_score": avg_score,
                "_semantic_score": 0.0,
                "_source": "keyword"
            })

    matches.sort(key=lambda x: -x["_keyword_score"])
    return matches[:top_k]

def keyword_retrieve_smart_fuzzy(query: str, paragraphs: Optional[List[Dict]] = None, top_k: int = 5) -> List[Dict]:
    """智能关键词检索"""
    if _client is None:
        raise RuntimeError("请先调用initialize()进行初始化")
        
    keywords = extract_keywords_gpt(query)
    print(f"GPT抽取关键词: {keywords}")

    variant_map = expand_keywords_gpt(keywords)
    print("同义词扩展:")
    for k, v in variant_map.items():
        print(f"- {k}: {v}")

    all_keywords = list(set(keywords + sum(variant_map.values(), [])))
    print(f"匹配关键词总数: {len(all_keywords)}")

    return keyword_match_fuzzy(all_keywords, paragraphs, top_k=top_k)

