# generator/answer_generator.py
"""
提供基于检索的问答生成功能。

此模块提供了一个回答生成器，可以根据检索到的相关上下文生成回答。
主要功能：
1. 初始化检索器和回答生成器
2. 支持多种模型提供商（OpenAI, Anthropic, DeepSeek等）
3. 根据用户查询生成有针对性的回答

使用示例：
    from generator import initialize, generate_answer, set_api_key
    
    # 初始化
    initialize(
        openai_api_key="your_openai_api_key",
        pinecone_api_key="your_pinecone_api_key",
        pinecone_index_name="your_index_name",
        json_path="./data/all_paragraphs.json"
    )
    
    # 设置其他模型提供商的API密钥
    set_api_key("anthropic", "your_anthropic_api_key")
    
    # 使用OpenAI生成回答
    answer1 = generate_answer(
        query="唐僧师徒经历了哪些劫难？",
        model_provider="openai",
        model_name="gpt-4o"
    )
    
    # 使用Anthropic生成回答
    answer2 = generate_answer(
        query="唐僧师徒经历了哪些劫难？",
        model_provider="anthropic",
        model_name="claude-3-opus-20240229"
    )
"""
import json
import os
import time
from typing import List, Dict, Tuple, Optional

# 导入检索器模块
from retriever.combined_retriever import combined_retrieve, initialize as initialize_combined

class AnswerGenerator:
    """回答生成器类，用于生成基于检索结果的回答"""
    
    def __init__(self):
        """初始化回答生成器"""
        self._initialized = False
        self._api_keys = {}  # 存储不同模型提供商的API密钥
    
    def initialize(
        self,
        openai_api_key: str,
        pinecone_api_key: str,
        pinecone_index_name: str,
        json_path: str,
        chat_model: str = "gpt-4o",
        anthropic_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None
    ) -> None:
        """
        初始化回答生成器
        
        Args:
            openai_api_key: OpenAI API密钥
            pinecone_api_key: Pinecone API密钥
            pinecone_index_name: Pinecone索引名称
            json_path: 段落数据的JSON文件路径
            chat_model: 使用的chat模型
            anthropic_api_key: Anthropic API密钥（可选）
            deepseek_api_key: DeepSeek API密钥（可选）
        """
        print("初始化回答生成器...")
        
        # 初始化组合检索器
        initialize_combined(
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key,
            pinecone_index_name=pinecone_index_name,
            json_path=json_path,
            chat_model=chat_model
        )
        
        # 保存各提供商的API密钥
        self._api_keys = {
            "openai": openai_api_key,
            "pinecone": pinecone_api_key
        }
        
        # 如果提供了其他API密钥，也保存下来
        if anthropic_api_key:
            self._api_keys["anthropic"] = anthropic_api_key
        if deepseek_api_key:
            self._api_keys["deepseek"] = deepseek_api_key
            
        self._initialized = True
        print("初始化完成！")
    
    def set_api_key(self, provider: str, api_key: str) -> None:
        """
        设置特定提供商的API密钥
        
        Args:
            provider: 模型提供商名称（如"openai", "anthropic", "deepseek"等）
            api_key: API密钥
        """
        self._api_keys[provider] = api_key
        print(f"已设置 {provider} 的API密钥")
        
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        获取特定提供商的API密钥
        
        Args:
            provider: 模型提供商名称
            
        Returns:
            API密钥，如果没有找到则返回None
        """
        return self._api_keys.get(provider)
    
    def build_prompt(self, query: str, max_context_chars: int = 50000) -> Tuple[str, List[Dict]]:
        """
        构建用于GPT问答的prompt
        
        Args:
            query: 用户查询
            max_context_chars: 最大上下文字符数
            
        Returns:
            构建好的prompt和检索到的段落
        """
        if not self._initialized:
            raise RuntimeError("请先调用initialize()进行初始化")
            
        # 使用combined_retrieve获取相关段落
        retrieved = combined_retrieve(query)
        
        # 构建上下文
        context = ""
        total_length = 0
        for para in retrieved:
            block = f"[第 {para['chapter_num']} 回] {para['chapter']}\n{para['text'].strip()}\n\n"
            if total_length + len(block) > max_context_chars:
                break
            context += block
            total_length += len(block)

        # # 构建完整prompt
        # prompt = (
        #     f"请根据以下《西游记》原文段落，回答用户的问题。\n"
        #     f"如果原文中提供了明确信息，请引用段落进行解释。\n"
        #     f"在引用时，请尽量列出与答案直接相关的原文段落片段内容（一到两句话）及对应章节。\n"
        #     f"如果用户要求你列举相关内容，你应列出所有可能的答案，不要仅列出部分答案或示例。。\n"
        #     f"如果信息不明确，请坦率说明无法确认，切勿编造。\n"
        #     f"---\n"
        #     f"{context}"
        #     f"---\n"
        #     f"用户问题：{query}\n"
        #     f"请用结构清晰的方式作答："
        # )
        prompt = (
            f"请根据以下《西游记》原文段落和你自身的知识库，回答用户的问题。\n"
            f"你的回答需要优先基于以下提供的原文内容，适当的使用你的背景知识。\n"
            f"如果原文中提供了明确信息，请引用段落进行解释，引用格式为：【第X回】原文内容。\n"
            f"在引用时，请尽量列出与答案直接相关的原文段落片段内容（一到两句话）及对应章节。\n"
            f"如果用户要求你列举相关内容，你应列出所有可能的答案，不要仅列出部分答案或示例。\n"
            f"如果信息不明确或原文中没有提供，请坦率说明无法确认，切勿编造。\n"
            f"---\n"
            f"{context}"
            f"---\n"
            f"用户问题：{query}\n"
            f"请用结构清晰的方式作答，尽可能使用原文中的信息："
        )
        
        return prompt, retrieved
    
    def get_client(self, model_provider: str, api_key: str):
        """
        获取不同模型提供商的客户端
        
        Args:
            model_provider: 模型提供商，支持"openai"、"anthropic"、"deepseek"
            api_key: API密钥
            
        Returns:
            相应的客户端实例
        """
        if model_provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=api_key)
        elif model_provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic(api_key=api_key)
        elif model_provider == "deepseek":
            from openai import OpenAI
            return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        else:
            raise ValueError(f"不支持的模型提供方：{model_provider}")
    
    def generate_answer(
        self,
        query: str,
        model_provider: str = "openai",
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_context_chars: int = 50000
    ) -> str:
        """
        生成回答
        
        Args:
            query: 用户查询
            model_provider: 模型提供商，默认为"openai"
            model_name: 模型名称，默认为"gpt-4o"
            api_key: API密钥，如果为None则使用初始化时保存的相应提供商的API密钥
            max_context_chars: 最大上下文字符数
            
        Returns:
            生成的回答
        
        Raises:
            RuntimeError: 如果尚未初始化或者缺少所需的API密钥
        """
        if not self._initialized:
            raise RuntimeError("请先调用initialize()进行初始化")
        
        # 如果未提供API密钥，则使用初始化时保存的相应提供商的API密钥
        if api_key is None:
            api_key = self.get_api_key(model_provider)
            if api_key is None:
                raise RuntimeError(f"未找到 {model_provider} 的API密钥，请在初始化时提供或通过set_api_key设置")
        
        # 构建prompt
        prompt, retrieved = self.build_prompt(query, max_context_chars)
        
        # 获取客户端
        client = self.get_client(model_provider, api_key)
        
        # 根据不同的模型提供商生成回答
        if model_provider == "openai":
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "你是一个严谨的《西游记》分析助手"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            answer = response.choices[0].message.content.strip()
        
        elif model_provider == "anthropic":
            response = client.messages.create(
                model=model_name,
                system="你是一个严谨的《西游记》分析助手",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            answer = response.content[0].text.strip()
        
        elif model_provider == "deepseek":
            import threading
            import queue
            from openai import OpenAI
            
            # 创建一个队列用于存储结果
            result_queue = queue.Queue()
            error_queue = queue.Queue()
            
            # 定义一个在单独线程中调用DeepSeek API的函数
            def call_deepseek_api():
                try:
                    # 记录开始时间
                    start_time = time.time()
                    print(f"开始调用DeepSeek {model_name} 模型...")
                    
                    # 创建客户端并调用API
                    client_local = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                    response = client_local.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "你是一个严谨的《西游记》分析助手"},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    # 将结果放入队列
                    elapsed = time.time() - start_time
                    print(f"DeepSeek模型响应成功，耗时 {elapsed:.2f} 秒")
                    result_queue.put(response.choices[0].message.content.strip())
                except Exception as e:
                    # 如果发生错误，将错误放入队列
                    error_queue.put(str(e))
                    print(f"DeepSeek模型调用失败: {str(e)}")
            
            # 创建并启动线程
            api_thread = threading.Thread(target=call_deepseek_api)
            api_thread.daemon = True  # 设置为守护线程，这样主线程退出时它会自动终止
            api_thread.start()
            
            # 等待结果，最多等待300秒
            try:
                # 每5秒检查一次并输出状态信息，以保持进程活动
                max_wait_time = 300  # 最长等待时间(秒)
                wait_interval = 5    # 检查间隔(秒)
                wait_count = 0
                
                while wait_count < max_wait_time:
                    # 检查是否有结果
                    try:
                        answer = result_queue.get(block=False)
                        break  # 有结果，退出等待循环
                    except queue.Empty:
                        pass  # 队列为空，继续等待
                        
                    # 检查是否有错误
                    try:
                        error_msg = error_queue.get(block=False)
                        raise RuntimeError(f"DeepSeek模型调用失败: {error_msg}")
                    except queue.Empty:
                        pass  # 没有错误，继续等待
                        
                    # 输出状态信息，保持进程活动
                    print(f"等待DeepSeek模型响应中，已等待 {wait_count} 秒...")
                    time.sleep(wait_interval)
                    wait_count += wait_interval
                    
                # 如果超过等待时间仍无响应
                if wait_count >= max_wait_time:
                    raise TimeoutError("DeepSeek模型响应超时(300秒)")
            
            except (TimeoutError, RuntimeError) as e:
                # 处理超时或API错误
                answer = f"很抱歉，使用DeepSeek模型({model_name})时遇到了问题: {str(e)}。\n\n建议您切换到GPT-4o模型以获得更快的响应。"
        
        else:
            raise ValueError(f"不支持的模型提供方：{model_provider}")
        
        return answer

# 创建全局实例以便直接导入使用
generator = AnswerGenerator()

# 导出便捷函数
def initialize(
    openai_api_key: str,
    pinecone_api_key: str,
    pinecone_index_name: str,
    json_path: str,
    chat_model: str = "gpt-4o",
    anthropic_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None
) -> None:
    """初始化回答生成器的便捷函数"""
    generator.initialize(
        openai_api_key=openai_api_key,
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        json_path=json_path,
        chat_model=chat_model,
        anthropic_api_key=anthropic_api_key,
        deepseek_api_key=deepseek_api_key
    )

def generate_answer(
    query: str,
    model_provider: str = "openai",
    model_name: str = "gpt-4o",
    api_key: Optional[str] = None,
    max_context_chars: int = 50000
) -> str:
    """生成回答的便捷函数"""
    return generator.generate_answer(
        query=query,
        model_provider=model_provider,
        model_name=model_name,
        api_key=api_key,
        max_context_chars=max_context_chars
    )

def set_api_key(provider: str, api_key: str) -> None:
    """
    设置特定提供商的API密钥的便捷函数
    
    Args:
        provider: 模型提供商名称（如"openai", "anthropic", "deepseek"等）
        api_key: API密钥
    """
    generator.set_api_key(provider, api_key) 