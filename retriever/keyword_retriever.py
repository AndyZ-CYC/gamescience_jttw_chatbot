# retriever/keyword_retriever.py
import re
import json
import logging
import time
from typing import List, Dict, Optional, Set, Tuple
from openai import OpenAI
from rapidfuzz import fuzz
import os

# 配置日志
logger = logging.getLogger("xiyouji-keyword-retriever")

# 全局配置
class Config:
    openai_api_key: Optional[str] = None
    chat_model: str = "gpt-4o"

# 全局客户端实例
_client: Optional[OpenAI] = None
_paragraphs: List[Dict] = []

def initialize(
    openai_api_key: str, 
    chat_model: str = "gpt-4o",
    paragraphs_file: Optional[str] = None
) -> None:
    """
    初始化关键词检索器
    
    Args:
        openai_api_key: OpenAI API密钥
        chat_model: 使用的chat模型
        paragraphs_file: 段落数据文件路径
    """
    global _client, _paragraphs
    
    # 更新配置
    Config.openai_api_key = openai_api_key
    Config.chat_model = chat_model
    
    # 初始化客户端
    try:
        logger.info("正在初始化OpenAI客户端...")
        _client = OpenAI(api_key=openai_api_key)
        logger.info("初始化完成!")
        
        # 加载段落数据(如果提供)
        if paragraphs_file and os.path.exists(paragraphs_file):
            logger.info(f"正在加载段落数据: {paragraphs_file}")
            load_paragraphs(paragraphs_file)
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}", exc_info=True)
        raise

def load_paragraphs(file_path: str) -> None:
    """加载段落数据"""
    global _paragraphs
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            _paragraphs = json.load(f)
        logger.info(f"成功加载 {len(_paragraphs)} 个段落")
    except Exception as e:
        logger.error(f"加载段落数据失败: {str(e)}", exc_info=True)
        raise

def extract_keywords_gpt(text: str, max_keywords: int = 6) -> List[str]:
    """使用GPT提取关键词"""
    if _client is None:
        raise RuntimeError("请先调用initialize()进行初始化")
    
    # 添加重试逻辑
    max_retries = 3
    for attempt in range(max_retries):
        try:
            system_prompt = (
                "你是一个精确的关键词提取助手。请从用户的问题中提取最重要的关键词，"
                "这些关键词将用于检索相关文献。请只返回关键词列表，每个关键词用逗号分隔，不要添加任何额外的文本或解释。"
                "提取的关键词应该保持原始形式，不要添加引号或其他标记。"
                "对于人名、地名、专有名词等，应完整提取。对于一般名词，提取最有意义的部分。"
                "例如：从'孙悟空为什么会被压在五指山下'中提取：孙悟空,五指山"
            )
            
            response = _client.chat.completions.create(
                model=Config.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请从以下问题中提取最多{max_keywords}个关键词：{text}"}
                ],
                temperature=0.2
            )
            
            keywords = [k.strip() for k in response.choices[0].message.content.split(",") if k.strip()]
            logger.info(f"从查询中提取了关键词: {keywords}")
            return keywords
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"关键词提取失败，尝试重试 ({attempt+1}/{max_retries}): {str(e)}")
                time.sleep(1 * (attempt + 1))  # 指数级退避
            else:
                logger.error(f"关键词提取失败: {str(e)}", exc_info=True)
                # 失败时返回简单分词结果作为后备方案
                simple_keywords = [w for w in re.findall(r'\w+', text) if len(w) > 1]
                logger.info(f"使用简单分词作为后备方案: {simple_keywords[:max_keywords]}")
                return simple_keywords[:max_keywords]

def expand_keywords_with_synonyms(keywords: List[str]) -> Dict[str, List[str]]:
    """扩展关键词与其同义词"""
    if _client is None:
        raise RuntimeError("请先调用initialize()进行初始化")
    
    if not keywords:
        return {}
    
    try:
        system_prompt = (
            "你是一个同义词专家。请为每个关键词提供2-3个在《西游记》语境中的同义词或相关表达。"
            "只返回同义词，不要添加解释。格式应为JSON字典，键为原关键词，值为同义词列表。"
        )
        
        keywords_str = ", ".join(keywords)
        response = _client.chat.completions.create(
            model=Config.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请为以下关键词提供同义词：{keywords_str}"}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            # 确保每个原始关键词都在结果中
            for k in keywords:
                if k not in result:
                    result[k] = []
            logger.info(f"成功扩展同义词: {result}")
            return result
        except json.JSONDecodeError:
            logger.warning("同义词扩展结果无法解析为JSON，使用空结果")
            return {k: [] for k in keywords}
    except Exception as e:
        logger.error(f"扩展同义词失败: {str(e)}", exc_info=True)
        return {k: [] for k in keywords}

def fuzzy_keyword_match(paragraphs: List[Dict], keywords_with_synonyms: Dict[str, List[str]], 
                       threshold: int = 80) -> List[Tuple[Dict, float]]:
    """使用模糊匹配进行关键词检索"""
    if not paragraphs or not keywords_with_synonyms:
        return []
    
    results = []
    
    # 构建所有需要搜索的关键词集合
    all_keywords: Set[str] = set()
    for keyword, synonyms in keywords_with_synonyms.items():
        all_keywords.add(keyword)
        all_keywords.update(synonyms)
    
    for para in paragraphs:
        para_text = para.get("text", "")
        # 跳过空段落
        if not para_text:
            continue
            
        max_score = 0
        matched_keywords = 0
        
        # 对每个关键词进行匹配
        for keyword in all_keywords:
            # 直接匹配
            if keyword in para_text:
                matched_keywords += 1
                max_score += 100
                continue
                
            # 模糊匹配
            words = re.findall(r'\w+', para_text)
            best_word_score = 0
            
            for word in words:
                if len(word) > 1:  # 忽略单字符词
                    sim_score = fuzz.ratio(keyword, word)
                    best_word_score = max(best_word_score, sim_score)
            
            if best_word_score >= threshold:
                matched_keywords += 1
                max_score += best_word_score
        
        # 至少匹配一个关键词才计入结果
        if matched_keywords > 0:
            # 计算综合分数：匹配数量与匹配质量的加权平均
            relevance_score = (max_score / len(all_keywords)) * (matched_keywords / len(all_keywords))
            results.append((para, relevance_score))
    
    # 按相关性得分排序
    return sorted(results, key=lambda x: -x[1])

def keyword_retrieve_smart_fuzzy(query: str, paragraphs: Optional[List[Dict]] = None, top_k: int = 5) -> List[Dict]:
    """
    智能模糊关键词检索
    
    Args:
        query: 查询文本
        paragraphs: 段落列表，如果为None则使用全局_paragraphs
        top_k: 返回结果数量
        
    Returns:
        检索结果列表
    """
    if _client is None:
        raise RuntimeError("请先调用initialize()进行初始化")
    
    if paragraphs is None:
        if not _paragraphs:
            logger.error("未提供段落数据且全局段落列表为空")
            return []
        paragraphs = _paragraphs
    
    try:
        # 步骤1：提取关键词
        keywords = extract_keywords_gpt(query)
        if not keywords:
            logger.warning("未能从查询中提取到关键词")
            return []
        
        # 步骤2：扩展同义词
        keywords_with_synonyms = expand_keywords_with_synonyms(keywords)
        
        # 步骤3：进行模糊匹配
        matches = fuzzy_keyword_match(paragraphs, keywords_with_synonyms)
        
        # 步骤4：整理返回结果
        return [
            {
                "id": str(i),
                "score": float(score),
                "text": match.get("text", ""),
                "chapter": match.get("chapter", ""),
                "chapter_num": match.get("chapter_num", -1)
            }
            for i, (match, score) in enumerate(matches[:top_k])
        ]
    except Exception as e:
        logger.error(f"关键词检索失败: {str(e)}", exc_info=True)
        return []

