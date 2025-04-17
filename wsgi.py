"""
WSGI入口点 - 用于Railway部署
"""
import os
import sys
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("xiyouji-app")

# 添加项目根目录到系统路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# 检查数据文件是否存在
data_path = os.path.join(BASE_DIR, "data", "all_paragraphs.json")
if not os.path.exists(data_path):
    logger.error(f"数据文件不存在: {data_path}")
    logger.error("确保数据文件已上传到正确的位置")
else:
    logger.info(f"找到数据文件: {data_path}")

try:
    # 导入应用
    from frontend.app import app, setup
    
    # 初始化应用
    logger.info("开始初始化应用...")
    setup()
    logger.info("应用初始化完成!")
    
    # 导出应用对象
    application = app
except Exception as error:
    logger.error(f"应用初始化失败: {str(error)}", exc_info=True)
    from flask import Flask, jsonify
    fallback_app = Flask(__name__)
    
    error_message = str(error)
    
    @fallback_app.route('/')
    def error_page():
        return jsonify({
            "error": "应用初始化失败",
            "message": error_message,
            "status": "请检查服务器日志获取更多信息"
        }), 500
    
    application = fallback_app

# Railway特定配置
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 3000))  # Railway默认使用3000端口
    application.run(host='0.0.0.0', port=port, debug=False) 