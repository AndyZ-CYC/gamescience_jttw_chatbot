# 西游记智能问答系统

本项目是一个基于《西游记》原著文本的智能问答系统，采用了RAG（Retrieval-Augmented Generation）架构，结合了语义检索和关键词检索，通过主流大语言模型生成回答，并提供了简洁的Web用户界面。

## 项目架构

此项目主要由四个模块组成：

1. **文本数据清洗与数据库构建**：处理原始文本数据，分段存储，并建立向量数据库
2. **检索模块**：包含语义检索和关键词检索两种方法，以及它们的组合策略
3. **回答生成模块**：使用AI模型基于检索内容生成回答
4. **前端界面**：提供Web交互界面

## 模块详细说明

### 1. 文本数据清洗与数据库构建

#### 模块说明

此模块负责离线部分的文本数据清洗与向量数据库的构建，为后续语义检索做好基础。

#### 相关文件

- **RAG数据库构建.ipynb**：包含数据处理和数据库构建的全流程代码
- **data/all_paragraphs.json**：处理后的文本段落存储文件
- **data/西游记_UTF-8.txt**：粗略清洗（移除页码与字数统计）后经过UTF-8编码的原始文本
- **data/西游记.txt**：原始文本文件

#### 主要功能

- 文本清洗：去除不必要的标点和格式
- 文本分段：按照章节和段落进行合理分割
- 向量生成：使用OpenAI的Embedding模型(text-embedding-3-large)生成文本向量
- Pinecone数据库：建立和管理向量索引与云端向量数据库

#### 问题与后续开发方向

- 可尝试不同文本段落分隔方式（如改为自然段分割，或更改每个文本段落字数限制）
- 可尝试使用其他Embedding模型
- 此模块代码可以被部分复用，后续如果有构建更完善RAG数据库的需求（如包含多本文献），可进行适当更改与应用

### 2. 检索模块

#### 模块说明

此模块负责使用不同检索方法检索与用户问题相关的文本片段与相关段落/章节信息，以后续Prompt文本构建。

#### 相关文件和目录

- **retriever/**：检索模块的主目录
  - **__init__.py**：模块初始化和函数导出
  - **semantic_retriever.py**：语义检索实现
  - **keyword_retriever.py**：关键词检索实现
  - **combined_retriever.py**：组合检索策略实现
- **语义与关键词检索.ipynb**：检索方法的开发和测试笔记本

#### 主要功能

- **语义检索（Semantic Retriever）**：
  - 近义扩展：通过GPT-4o根据用户输入内容生成含义近似的表达方式，以增强语义检索的召回值(recall)
  - 多查询检索：将扩展后的多个查询结果合并
  
- **关键词检索（Keyword Retriever）**：
  - 关键词提取：使用GPT-4o从用户问题中提取关键词
  - 同义词扩展：使用GPT-4o对关键词进行同义词扩展
  - 模糊匹配（fuzz）：使用模糊匹配算法匹配文本段落
  
- **组合检索**：
  - 参数选择：根据用户问题所含关键词判断是否需要列举全部情况，半自动调整top_k参数
  - 结果融合：智能融合语义和关键词检索结果
  - 排序优化：根据相关性重新排序检索结果，并返回前top_k个结果

#### 问题与后续开发方向

- 关键词检索在测试问题中的准确率通常略低于语义检索，并且目前关键词检索在组合检索中的权重也略低，但考虑到对某些特定问题会有奇效，后续可以进行权重调整和关键词分数加权的开发
- 目前检索使用的是RAG的基本逻辑，而Rerank模型在某些情况表现会优于普通的RAG，但考虑到《西游记》单本小说的样本量较低、Rerank模型时间成本偏大等问题，暂时没有采用
- RAG在匹配用户的某些抽象问题（比如"列举所有一打多的场景"）时效果不佳，因此如果想提升这类问题的精度可能需要更完善的数据库或者更精细的标签

### 3. 回答生成模块

#### 功能说明

此模块为项目核心模块，基于检索内容生成连贯、准确的回答。开发架构支持多种不同大语言模型供用户选择，并提供一定背景、要求与预设来增强回答效果。

#### 相关文件和目录

- **generator/**：回答生成模块的主目录
  - **__init__.py**：模块初始化和函数导出
  - **answer_generator.py**：答案生成器实现
- **回答生成.ipynb**：生成方法的开发和测试笔记本

#### 主要功能

- 初始化配置：设置API密钥和模型参数
- Prompt构建：根据检索结果构建有效的Prompt文本
- 回答生成：调用大语言模型生成回答

#### 问题与后续开发方向

- 目前代码只支持openai，deepseek，和anthropic三个开发商的模型，可以通过修改部分代码模块添加更多可支持的模型

### 4. 前端UI模块

#### 模块说明

此模块负责提供简介的的Web界面，支持模型选择和问题输入，并实时显示回答和状态

#### 相关文件和目录

- **frontend/**：前端模块的主目录
  - **app.py**：Flask应用主文件
  - **run.py**：启动脚本
  - **README.md**：前端使用说明
  - **static/**：静态资源目录
    - **css/style.css**：样式表
    - **js/app.js**：前端JavaScript代码
  - **templates/**：
    - **index.html**：HTML模板

#### 主要功能

- 用户界面：简洁直观的聊天界面
- 多模型支持：可选择不同的生成模型
- 响应式设计：适配不同设备屏幕
- Markdown渲染：支持格式化文本输出
- 状态指示：显示系统处理状态

#### 后续开发方向

- 添加更多功能，比如支持用户自行输入API密钥
- 美化界面，添加外部图片素材等

## 运行指南

### 方法1：本地运行

##### 准备环境

1. 安装Python 3.8+
2. 安装依赖包：
   ```
   pip install -r requirements.txt
   ```
3. 设置环境变量（选择以下任一方式）：

   方式1 - 使用环境变量：
   ```bash
   # Windows
   set OPENAI_API_KEY=你的OpenAI API密钥
   set PINECONE_API_KEY=你的Pinecone API密钥
   set DEEPSEEK_API_KEY=你的DeepSeek API密钥

   # Linux/Mac
   export OPENAI_API_KEY=你的OpenAI API密钥
   export PINECONE_API_KEY=你的Pinecone API密钥
   export DEEPSEEK_API_KEY=你的DeepSeek API密钥
   ```

   方式2 - 使用.env文件：
   在项目根目录创建 `.env` 文件：
   ```
   OPENAI_API_KEY=你的OpenAI API密钥
   PINECONE_API_KEY=你的Pinecone API密钥
   DEEPSEEK_API_KEY=你的DeepSeek API密钥
   PINECONE_INDEX_NAME=xiyouji-embedding
   ```

##### 启动系统

1. 启动前端服务：
   ```
   python wsgi.py
   ```
2. 访问Web界面：
   ```
   http://127.0.0.1:3000
   ```

### 方法2：使用Railway部署（推荐）

Railway.app 提供了最稳定的云端部署方案：

1. 在 Railway.app 创建新项目
2. 连接 GitHub 仓库
3. 设置环境变量：
   - 在 Railway 仪表板中添加必要的环境变量（同上述环境变量列表）
4. Railway 会自动检测并部署项目
5. 部署完成后，可以通过 Railway 提供的域名访问系统

**Railway.app 优势**：
- 稳定的运行环境
- 所有模型都能正常工作
- 自动化部署流程
- 合理的资源分配
- 较短的冷启动时间

### 方法3：其他部署选项

虽然系统也支持在其他平台（如Render.com）上部署，但由于性能限制和环境差异，可能会导致部分模型无法正常工作或经常超时。因此建议优先选择Railway.app或本地部署。

## 部署注意事项

### 环境变量配置

如果环境变量未设置，系统会使用以下默认值：
- OPENAI_API_KEY：无默认值，必须设置
- PINECONE_API_KEY：无默认值，必须设置
- DEEPSEEK_API_KEY：无默认值，可选
- PINECONE_INDEX_NAME：默认为 "xiyouji-embedding"
- PORT：Railway默认使用3000端口，Render默认使用5000端口

### 部署平台特性

- **Railway.app（推荐）**：
  - 使用 Nixpacks 构建系统
  - 默认端口为3000
  - 支持自动化部署
  - 所有模型均可正常使用
  - 适合中小规模应用
  - 稳定的运行环境

### 性能优化建议

1. 选择合适的部署方案：
   - 个人使用：本地部署
   - 团队/生产环境：Railway.app（推荐）
   - 其他平台：需注意模型兼容性和性能限制

2. 根据使用情况选择合适的实例规格：
   - Railway：默认配置通常足够满足需求

3. 模型选择建议：
   - **在Railway.app上**：
     - GPT-4o：默认推荐，响应最快
     - DeepSeek模型：适合中文语境，通常有不错的表现
     - GPT-4.5-preview：适合复杂问题，但费用较高
   - **在其他平台上**：
     - 建议仅使用GPT-4o模型以确保稳定性
     - 其他模型可能出现超时或性能问题

### 模型性能与超时说明

- **平台选择建议**：
  - Railway.app：所有模型均可正常使用
  - 其他平台：建议仅使用GPT-4o模型
  
- **模型特性**：
  - **GPT-4o**：最稳定可靠的选择，适合所有部署环境
  - **GPT-4.5-preview**：仅在Railway.app上表现稳定，其他平台可能频繁超时
  - **DeepSeek系列模型**：
    - 在Railway.app上：性能良好，适合中文问答
    - 在其他平台：可能出现超时或性能问题

- **使用建议**：
  - 如果使用Railway.app部署，可以自由选择任何模型
  - 如果使用其他平台部署，建议仅启用GPT-4o模型以避免用户体验问题
  - 对于高并发或需要稳定性的场景，推荐使用Railway.app部署

## 使用工具汇总

- **向量数据库**：Pinecone
- **嵌入模型**：OpenAI - text-embedding-3-large
- **生成模型**：OpenAI的GPT-4o，GPT-4.5-preview模型，Deepseek的V3，R1模型
- **后端框架**：Flask
- **前端技术**：HTML, CSS, JavaScript, Bootstrap

## 其他注意事项与问题

- 目前GPT-4.5-preview模型较为昂贵，虽然可能是表现最好的模型，但可能需要谨慎使用
- Deepseek-R1模型由于深度思索的特性所以响应时间较长
- 由于地区服务限制，暂时没有引进Anthropic的API密钥
- 目前本模型能够准确的引用西游记原文内容，并基本杜绝AI的幻觉问题（hallucination）。然而在进行列举工作的时候，仍有可能出现列举不完全的问题；与此同时，模型对一些抽象的问题回答准确性会有一定的降低。

### 模型性能与超时说明

- **推荐模型**：GPT-4o是系统默认模型，综合性能与速度最佳，推荐首选。
- **特殊需求模型**：
  - **GPT-4.5-preview**：对于更复杂的问题可能有更好表现，但响应时间较长，可能偶尔超时。
  - **DeepSeek系列模型**：响应时间不稳定，在网络波动大的情况下可能超时。
- **超时处理**：系统内置了自动容错机制，当非GPT-4o模型超时时，会自动回退到GPT-4o模型并给出提示。
- **大量内容检索**：提问时包含"列举"、"全部"、"八十一难"等关键词的问题会触发高召回率模式，这类问题建议使用GPT-4o模型以避免超时。

© 2025 张耀元 | 西游记问答系统 
