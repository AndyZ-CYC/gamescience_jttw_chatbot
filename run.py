"""
西游记问答系统启动脚本
运行方法：python run.py
"""
import os
import sys

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置控制台输出编码为UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 导入前端应用
from frontend.app import app, setup

if __name__ == "__main__":
    print("正在初始化西游记问答系统...")
    # 确保在启动前初始化模型
    setup()
    print("系统初始化完成！正在启动Web服务...")
    print("请在浏览器中访问：http://127.0.0.1:5000")
    app.run(debug=True)