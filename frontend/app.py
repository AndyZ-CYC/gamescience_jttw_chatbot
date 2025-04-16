import os
import sys
import time
from flask import Flask, render_template, request, jsonify

# 添加项目根目录到系统路径
if getattr(sys, 'frozen', False):
    # 在PyInstaller打包的环境中运行
    BASE_DIR = os.environ.get('APP_ROOT', sys._MEIPASS)
else:
    # 在普通Python环境中运行
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(BASE_DIR)

# 设置控制台输出编码
if hasattr(sys.stdout, 'reconfigure') and sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 导入生成器模块
from generator import initialize, generate_answer, set_api_key

# API密钥和配置
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', "")
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', "pcsk_5u46KS_Hx3E1L6rTJqeYJ7GmEmDEtpzT6juNNHBmVQTNEaoKr7uaH5tJHzjjdKcU3GaUZw")
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', "sk-e7852049ff9940a2b39113eeb97f1103")
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME', "xiyouji-embedding")

# 使用绝对路径定位JSON文件
JSON_PATH = os.path.join(BASE_DIR, "data", "all_paragraphs.json")
print(f"数据文件路径: {JSON_PATH}")

# 定义模型配置
MODELS = {
    "gpt-4o": {"provider": "openai", "name": "gpt-4o", "display_name": "GPT-4o"},
    "gpt-4.5-preview": {"provider": "openai", "name": "gpt-4.5-preview", "display_name": "GPT-4.5 Preview"},
    "deepseek-v3": {"provider": "deepseek", "name": "deepseek-chat", "display_name": "DeepSeek-V3"},
    "deepseek-r1": {"provider": "deepseek", "name": "deepseek-reasoner", "display_name": "DeepSeek-R1"}
}

# 初始化函数，在应用启动时调用
def setup():
    print("初始化回答生成器...")
    initialize(
        openai_api_key=OPENAI_API_KEY,
        pinecone_api_key=PINECONE_API_KEY,
        pinecone_index_name=PINECONE_INDEX_NAME,
        json_path=JSON_PATH
    )
    
    # 设置DeepSeek API密钥
    set_api_key("deepseek", DEEPSEEK_API_KEY)
    print("初始化完成！")

# 初始化应用
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 确保JSON响应中的非ASCII字符(如中文)不被转义

# 在应用上下文中添加一个初始化函数
@app.route('/init', methods=['GET'])
def init_app():
    setup()
    return jsonify({"status": "initialized"})

@app.route('/')
def index():
    return render_template('index.html', models=MODELS)

@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query', '')
    model_id = data.get('model', 'gpt-4o')
    
    if not query_text:
        return jsonify({"error": "请输入问题"}), 400
    
    if model_id not in MODELS:
        return jsonify({"error": "无效的模型"}), 400
    
    model_config = MODELS[model_id]
    
    try:
        # 设置合理的超时时间，避免请求被挂起太久
        start_time = time.time()
        
        # 对于非GPT-4o模型添加提示
        is_gpt4o = (model_config["provider"] == "openai" and model_config["name"] == "gpt-4o")
        warning_msg = "" if is_gpt4o else "注意：对于复杂问题，处理可能需要较长时间。若遇到问题，可尝试简化您的问题或稍后重试。"
        
        # 避免在render.com上超时的措施
        process_limit = 240  # 秒 - 增加时间限制
        
        answer = generate_answer(
            query=query_text,
            model_provider=model_config["provider"],
            model_name=model_config["name"]
        )
        
        processing_time = time.time() - start_time
        print(f"查询处理完成，耗时: {processing_time:.2f}秒")
        
        if warning_msg and processing_time > 15:  # 如果处理时间较长，加入警告提示
            answer = f"{warning_msg}\n\n{answer}"
            
        return jsonify({"answer": answer})
    except Exception as e:
        error_msg = str(e)
        print(f"生成回答时出错: {error_msg}")
        
        # 确保总是返回JSON响应，即使在错误情况下
        try:
            # 检查是否包含超时错误的典型特征
            if "timeout" in error_msg.lower() or ("time" in error_msg.lower() and "out" in error_msg.lower()):
                return jsonify({
                    "error": "处理请求超时",
                    "message": "您的查询处理时间过长，服务器响应超时。请尝试简化您的问题或稍后再试。"
                }), 408  # 408 Request Timeout
            else:
                return jsonify({
                    "error": "生成回答时出错", 
                    "message": f"处理出错: {str(e)[:200]}"
                }), 500
        except Exception as json_error:
            # 如果连JSON化错误信息都失败，返回最简单的JSON
            print(f"无法JSON化错误信息: {str(json_error)}")
            return jsonify({
                "error": "系统错误",
                "message": "处理请求时发生严重错误，请尝试刷新页面或稍后重试"
            }), 500

if __name__ == '__main__':
    # 在启动前初始化
    setup()
    # 获取端口，如果在环境变量中没有定义，则默认为5000
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port) 