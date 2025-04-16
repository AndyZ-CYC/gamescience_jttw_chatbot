"""
WSGI入口点 - 用于Gunicorn启动
"""
import os
import sys
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("xiyouji-app")

# 设置Gunicorn配置 (通过环境变量，这些会被wsgi文件读取但不生效，需在命令行或配置文件中设置)
os.environ.setdefault('GUNICORN_TIMEOUT', '300')  # 设置工作进程超时为300秒
os.environ.setdefault('GUNICORN_WORKERS', '2')    # 设置工作进程数
os.environ.setdefault('GUNICORN_KEEPALIVE', '5')  # 设置keepalive

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 检查数据文件是否存在
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "all_paragraphs.json")
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
    
    # 导出应用对象供Gunicorn使用
    application = app
except Exception as error:
    logger.error(f"应用初始化失败: {str(error)}", exc_info=True)
    # 创建一个最小的应用以显示错误信息
    from flask import Flask, jsonify
    fallback_app = Flask(__name__)
    
    # 保存错误信息，以便在路由函数中使用
    error_message = str(error)
    
    @fallback_app.route('/')
    def error_page():
        return jsonify({
            "error": "应用初始化失败",
            "message": error_message,
            "status": "请检查服务器日志获取更多信息"
        }), 500
    
    application = fallback_app

if __name__ == "__main__":
    # 直接运行此文件时
    port = int(os.environ.get('PORT', 5000))
    application.run(host='0.0.0.0', port=port) 