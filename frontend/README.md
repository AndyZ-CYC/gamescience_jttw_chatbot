# 西游记问答系统前端

这是西游记问答系统的Web前端界面，基于Flask实现。

## 功能特点

- 简洁直观的聊天界面
- 支持多种LLM模型（GPT-4o、GPT-4.5-Preview、DeepSeek-V3、DeepSeek-R1）
- Markdown格式回答的渲染
- 响应式设计，适配不同设备

## 运行方法

1. 确保已安装所需依赖：

```bash
pip install flask openai anthropic
```

2. 从项目根目录运行以下命令：

```bash
python frontend/run.py
```

3. 在浏览器中访问 http://127.0.0.1:5000

## 文件结构

```
frontend/
├── app.py            # Flask应用主文件
├── run.py            # 启动脚本
├── static/           # 静态资源
│   ├── css/          # 样式文件
│   └── js/           # JavaScript文件
└── templates/        # HTML模板
```

## 使用说明

1. 在页面上方的下拉菜单中选择要使用的模型
2. 在输入框中输入您关于《西游记》的问题
3. 点击发送按钮或按Enter键提交问题
4. 系统会根据《西游记》原著内容生成回答

## 可用模型

- **GPT-4o**: OpenAI的最新通用模型
- **GPT-4.5-Preview**: OpenAI的预览版模型
- **DeepSeek-V3**: DeepSeek的对话模型
- **DeepSeek-R1**: DeepSeek的深度思考模型

## 注意事项

- 首次启动时，系统需要初始化检索模型，可能需要一些时间
- 回答生成过程会根据问题复杂度和选用的模型有不同的延迟 