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
import logging
import traceback
from typing import List, Dict, Tuple, Optional, Any, Union

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('generator.log')
    ]
)
logger = logging.getLogger('answer_generator')

# 导入检索器模块
from retriever.combined_retriever import combined_retrieve, initialize as initialize_combined

class AnswerGenerator:
    """回答生成器类，用于生成基于检索结果的回答"""
    
    def __init__(self):
        """初始化回答生成器"""
        self._initialized = False
        self._api_keys = {}  # 存储不同模型提供商的API密钥
        self._clients = {}   # 缓存客户端实例
    
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
        logger.info("初始化回答生成器...")
        
        try:
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
            logger.info("回答生成器初始化完成！")
        except Exception as e:
            logger.error(f"回答生成器初始化失败: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"初始化失败: {str(e)}")
    
    def set_api_key(self, provider: str, api_key: str) -> None:
        """
        设置特定提供商的API密钥
        
        Args:
            provider: 模型提供商名称（如"openai", "anthropic", "deepseek"等）
            api_key: API密钥
        """
        self._api_keys[provider] = api_key
        # 当API密钥更新时，清除对应的客户端缓存
        if provider in self._clients:
            del self._clients[provider]
        logger.info(f"已设置 {provider} 的API密钥")
        
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        获取特定提供商的API密钥
        
        Args:
            provider: 模型提供商名称
            
        Returns:
            API密钥，如果没有找到则返回None
        """
        key = self._api_keys.get(provider)
        if key is None:
            logger.warning(f"未找到 {provider} 的API密钥")
        return key
    
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
        
        logger.info(f"为查询构建prompt: '{query}'")
        
        try:    
            # 使用combined_retrieve获取相关段落
            start_time = time.time()
            retrieved = combined_retrieve(query)
            logger.debug(f"检索完成，耗时 {time.time() - start_time:.2f}秒，获取到 {len(retrieved)} 个段落")
            
            # 构建上下文
            context = ""
            total_length = 0
            used_paragraphs = 0
            
            for para in retrieved:
                block = f"[第 {para['chapter_num']} 回] {para['chapter']}\n{para['text'].strip()}\n\n"
                if total_length + len(block) > max_context_chars:
                    break
                context += block
                total_length += len(block)
                used_paragraphs += 1

            logger.debug(f"使用了 {used_paragraphs} 个段落，总字符数 {total_length}")

            # 构建完整prompt
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
        except Exception as e:
            logger.error(f"构建prompt失败: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"构建prompt失败: {str(e)}")
    
    def get_client(self, model_provider: str, api_key: str):
        """
        获取不同模型提供商的客户端，使用缓存提高性能
        
        Args:
            model_provider: 模型提供商，支持"openai"、"anthropic"、"deepseek"
            api_key: API密钥
            
        Returns:
            相应的客户端实例
        """
        # 防止客户端复用导致的流读取问题，不再使用缓存机制
        # 每次请求都创建新客户端实例
        try:
            import httpx
            
            # 创建自定义的httpx客户端，设置超时时间为10分钟
            httpx_client = httpx.Client(
                timeout=httpx.Timeout(
                    connect=60.0,       # 连接超时
                    read=600.0,         # 读取超时
                    write=60.0,         # 写入超时
                    pool=60.0           # 连接池超时
                ),
                limits=httpx.Limits(
                    max_keepalive_connections=10,
                    max_connections=50,
                    keepalive_expiry=60.0
                )
            )
            
            if model_provider == "openai":
                from openai import OpenAI
                client = OpenAI(
                    api_key=api_key,
                    http_client=httpx_client
                )
                logger.debug("创建了新的OpenAI客户端")
                return client
            elif model_provider == "anthropic":
                from anthropic import Anthropic
                client = Anthropic(api_key=api_key)
                logger.debug("创建了新的Anthropic客户端")
                return client
            elif model_provider == "deepseek":
                from openai import OpenAI
                client = OpenAI(
                    api_key=api_key, 
                    base_url="https://api.deepseek.com",
                    http_client=httpx_client
                )
                logger.debug("创建了新的DeepSeek客户端")
                return client
            else:
                raise ValueError(f"不支持的模型提供方：{model_provider}")
            
        except ImportError as e:
            logger.error(f"导入{model_provider}客户端库失败: {str(e)}")
            raise RuntimeError(f"无法使用{model_provider}：所需库未安装。错误: {str(e)}")
        except Exception as e:
            logger.error(f"创建{model_provider}客户端失败: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"创建{model_provider}客户端失败: {str(e)}")
    
    def call_model_with_retry(
        self, 
        client: Any, 
        model_provider: str, 
        model_name: str, 
        prompt: str,
        max_retries: int = 2,  # 减少重试次数以避免长时间运行
        retry_delay: float = 1.5  # 减少重试延迟
    ) -> str:
        """
        调用模型生成回答，带重试机制
        
        Args:
            client: 模型客户端实例
            model_provider: 模型提供商
            model_name: 模型名称
            prompt: 提示词
            max_retries: 最大重试次数
            retry_delay: 重试延迟时间（秒）
            
        Returns:
            生成的回答
        """
        retries = 0
        last_error = None
        start_total_time = time.time()  # 添加总时间跟踪
        
        # 移除超时时间设置，依赖底层HTTP库的默认超时
        # 不再需要这段代码
        # timeout_seconds = 60  # 增加默认超时时间
        # if model_name in ["gpt-4.5-preview", "deepseek-reasoner"]:
        #     logger.info(f"使用扩展超时时间(90秒)的模型: {model_name}")
        #     timeout_seconds = 90
        # elif model_provider != "openai" or model_name != "gpt-4o":
        #     logger.info(f"使用非GPT-4o模型，增加超时时间至75秒: {model_provider}/{model_name}")
        #     timeout_seconds = 75
            
        while retries <= max_retries:
            try:
                if retries > 0:
                    logger.info(f"尝试第 {retries} 次重试...")
                    time.sleep(retry_delay * (2 ** (retries - 1)))  # 指数退避策略
                    
                    # 重试时重新创建客户端，避免连接复用导致的问题
                    if model_provider == "openai":
                        from openai import OpenAI
                        client = OpenAI(api_key=self.get_api_key("openai"))
                    elif model_provider == "deepseek":
                        from openai import OpenAI
                        client = OpenAI(api_key=self.get_api_key("deepseek"), base_url="https://api.deepseek.com")
                
                call_start_time = time.time()  # 修复变量名
                
                # 为长文本截断过长的prompt
                truncated_prompt = self._truncate_prompt_if_needed(prompt, model_name)
                if truncated_prompt != prompt:
                    logger.info(f"Prompt已截断，原长度: {len(prompt)}，截断后: {len(truncated_prompt)}")
                
                if model_provider == "openai":
                    try:
                        # 移除超时参数
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "你是一个严谨的《西游记》分析助手"},
                                {"role": "user", "content": truncated_prompt}
                            ],
                            temperature=0.3
                            # 移除timeout参数
                        )
                        answer = response.choices[0].message.content.strip()
                    except Exception as api_error:
                        logger.warning(f"OpenAI API调用出错: {str(api_error)}")
                        # 如果不是GPT-4o，尝试回退到GPT-4o
                        if model_name != "gpt-4o" and retries >= max_retries - 1:
                            from openai import OpenAI
                            logger.warning(f"尝试使用GPT-4o作为后备模型")
                            # 每次都创建新客户端，避免连接复用问题
                            backup_client = OpenAI(api_key=self.get_api_key("openai"))
                            backup_response = backup_client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": "你是一个严谨的《西游记》分析助手"},
                                    {"role": "user", "content": truncated_prompt}
                                ],
                                temperature=0.3
                                # 移除timeout参数
                            )
                            answer = backup_response.choices[0].message.content.strip()
                            answer = f"[注: 原选择的模型({model_name})响应超时，系统自动切换到GPT-4o模型]\n\n{answer}"
                        else:
                            raise
                
                elif model_provider == "anthropic":
                    response = client.messages.create(
                        model=model_name,
                        system="你是一个严谨的《西游记》分析助手",
                        messages=[{"role": "user", "content": truncated_prompt}],
                        temperature=0.3
                        # 移除timeout参数
                    )
                    answer = response.content[0].text.strip()
                
                elif model_provider == "deepseek":
                    try:
                        # DeepSeek API可能较慢
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "你是一个严谨的《西游记》分析助手"},
                                {"role": "user", "content": truncated_prompt}
                            ]
                            # 移除timeout参数
                        )
                        answer = response.choices[0].message.content.strip()
                    except Exception as api_error:
                        logger.warning(f"DeepSeek API调用出错: {str(api_error)}")
                        # 最后一次尝试，使用OpenAI GPT-4o作为后备
                        if retries >= max_retries - 1:
                            logger.warning(f"DeepSeek模型失败，切换到GPT-4o")
                            backup_client = OpenAI(api_key=self.get_api_key("openai"))
                            backup_response = backup_client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": "你是一个严谨的《西游记》分析助手"},
                                    {"role": "user", "content": truncated_prompt}
                                ],
                                temperature=0.3
                                # 移除timeout参数
                            )
                            answer = backup_response.choices[0].message.content.strip()
                            answer = f"[注: DeepSeek模型({model_name})响应超时，系统自动切换到GPT-4o模型]\n\n{answer}"
                        else:
                            raise
                else:
                    raise ValueError(f"不支持的模型提供方：{model_provider}")
                
                call_duration = time.time() - call_start_time
                logger.info(f"模型响应生成完成，耗时 {call_duration:.2f}秒")
                
                # 检查总处理时间作为警告日志
                total_time = time.time() - start_total_time
                if total_time > 150:  # 2.5分钟以上记录警告
                    logger.warning(f"处理时间较长，已用时 {total_time:.2f}秒")
                
                return answer
                
            except Exception as e:
                last_error = e
                retries += 1
                error_str = str(e)
                # 记录详细错误信息以便排查
                if "body stream already read" in error_str:
                    logger.error(f"HTTP响应流已读取错误: {error_str}")
                    # 此类错误需要重建客户端
                    self._clients = {}  # 清空客户端缓存
                    
                logger.warning(f"调用{model_provider}模型失败 (尝试 {retries}/{max_retries}): {error_str}")
                    
                if retries > max_retries:
                    break
        
        logger.error(f"在 {max_retries} 次尝试后仍无法生成回答: {str(last_error)}")
            
        # 生成简单的备用回答
        return f"很抱歉，我暂时无法回答您的问题。系统遇到了技术问题：{str(last_error)[:100]}...\n\n您可以尝试稍后再试或简化您的问题。"
        
    def _truncate_prompt_if_needed(self, prompt: str, model_name: str) -> str:
        """截断过长的prompt以避免超出模型最大上下文长度或超时"""
        # 根据不同模型设置不同的最大长度
        max_length = 64000  # 默认值
        
        if model_name == "gpt-4.5-preview":
            max_length = 64000
        elif model_name == "gpt-4o":
            max_length = 64000
        elif "deepseek" in model_name:
            max_length = 64000
            
        if len(prompt) <= max_length:
            return prompt
            
        # 保留前后内容，截断中间
        # 分割prompt以保留重要部分
        parts = prompt.split("---")
        if len(parts) >= 3:
            # 保留指令(第一部分)和问题(最后部分)
            instructions = parts[0]
            context_raw = "---" + parts[1] + "---"
            question = parts[2]
            
            # 计算需要保留的上下文长度
            available_length = max_length - len(instructions) - len(question) - 100
            if available_length <= 0:
                # 极端情况，只保留指令和问题
                return instructions + "---" + "上下文过长，已省略部分内容" + "---" + question
                
            # 截断上下文
            truncated_context = context_raw[:available_length // 2] + "\n...\n[内容过长已省略中间部分]...\n" + context_raw[-available_length // 2:]
            return instructions + truncated_context + question
        
        # 如果不符合预期格式，简单截断
        return prompt[:max_length // 2] + "\n...\n[内容过长已省略中间部分]...\n" + prompt[-max_length // 2:]
    
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
            logger.error("尝试在初始化前生成回答")
            raise RuntimeError("请先调用initialize()进行初始化")
        
        logger.info(f"处理用户查询: '{query}'，使用模型: {model_provider}/{model_name}")
        
        try:
            # 如果未提供API密钥，则使用初始化时保存的相应提供商的API密钥
            if api_key is None:
                api_key = self.get_api_key(model_provider)
                if api_key is None:
                    logger.error(f"未找到 {model_provider} 的API密钥")
                    raise RuntimeError(f"未找到 {model_provider} 的API密钥，请在初始化时提供或通过set_api_key设置")
            
            # 构建prompt
            prompt, retrieved = self.build_prompt(query, max_context_chars)
            
            # 获取客户端
            client = self.get_client(model_provider, api_key)
            
            # 调用模型生成回答（带重试）
            answer = self.call_model_with_retry(client, model_provider, model_name, prompt)
            
            # 简单检查生成的回答是否有效
            if not answer or len(answer.strip()) < 10:
                logger.warning("生成的回答过短或为空")
                answer = "很抱歉，我无法针对您的问题生成有效回答。请尝试重新表述您的问题，或稍后再试。"
            
            return answer
            
        except Exception as e:
            logger.error(f"生成回答时发生错误: {str(e)}")
            logger.debug(traceback.format_exc())
            return f"很抱歉，处理您的问题时遇到了技术难题：{str(e)[:100]}..."

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
    try:
        generator.initialize(
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key,
            pinecone_index_name=pinecone_index_name,
            json_path=json_path,
            chat_model=chat_model,
            anthropic_api_key=anthropic_api_key,
            deepseek_api_key=deepseek_api_key
        )
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}")
        raise

def generate_answer(
    query: str,
    model_provider: str = "openai",
    model_name: str = "gpt-4o",
    api_key: Optional[str] = None,
    max_context_chars: int = 50000
) -> str:
    """生成回答的便捷函数"""
    try:
        return generator.generate_answer(
            query=query,
            model_provider=model_provider,
            model_name=model_name,
            api_key=api_key,
            max_context_chars=max_context_chars
        )
    except Exception as e:
        logger.error(f"生成回答失败: {str(e)}")
        return f"很抱歉，系统暂时无法处理您的请求: {str(e)[:100]}..."

def set_api_key(provider: str, api_key: str) -> None:
    """
    设置特定提供商的API密钥的便捷函数
    
    Args:
        provider: 模型提供商名称（如"openai", "anthropic", "deepseek"等）
        api_key: API密钥
    """
    generator.set_api_key(provider, api_key) 