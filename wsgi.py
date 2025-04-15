"""
WSGI入口点 - 用于Gunicorn启动
"""
import os
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入应用并初始化
from frontend.app import app, setup

# 初始化应用
setup()

# 导出应用对象供Gunicorn使用
application = app

if __name__ == "__main__":
    # 直接运行此文件时
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 