# retriever/__init__.py
from .semantic_retriever import (
    initialize as initialize_semantic,
    get_query_embedding,
    expand_query_semantically,
    semantic_retrieve_multi_query,
    smart_semantic_retrieve
)

from .keyword_retriever import (
    initialize as initialize_keyword,
    load_paragraphs,
    extract_keywords_gpt,
    expand_keywords_gpt,
    keyword_match_fuzzy,
    keyword_retrieve_smart_fuzzy
)

from .combined_retriever import combined_retrieve

def initialize(
    openai_api_key: str,
    pinecone_api_key: str,
    pinecone_index_name: str = "xiyouji-embedding",
    embedding_model: str = "text-embedding-3-large",
    chat_model: str = "gpt-4o"
) -> None:
    """
    初始化检索器
    
    Args:
        openai_api_key: OpenAI API密钥
        pinecone_api_key: Pinecone API密钥
        pinecone_index_name: Pinecone索引名称
        embedding_model: 使用的embedding模型
        chat_model: 使用的chat模型
    """
    # 初始化语义检索器
    initialize_semantic(
        openai_api_key=openai_api_key,
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        embedding_model=embedding_model,
        chat_model=chat_model
    )
    
    # 初始化关键词检索器
    initialize_keyword(
        openai_api_key=openai_api_key,
        chat_model=chat_model
    )

__all__ = [
    # 初始化函数
    'initialize',
    
    # semantic retriever
    'get_query_embedding',
    'expand_query_semantically',
    'semantic_retrieve_multi_query',
    'smart_semantic_retrieve',
    
    # keyword retriever
    'load_paragraphs',
    'extract_keywords_gpt',
    'expand_keywords_gpt',
    'keyword_match_fuzzy',
    'keyword_retrieve_smart_fuzzy',
    
    # combined retriever
    'combined_retrieve'
] 