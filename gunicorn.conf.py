# gunicorn.conf.py - Gunicorn服务器配置文件

# 工作进程超时时间（秒）
# 这个值应该足够大，以容纳最长的模型响应时间
timeout = 300  # 5分钟

# 工作进程数
workers = 2

# 工作进程类型
worker_class = 'gevent'  # 使用gevent异步工作模式

# 每个工作进程的最大连接数
worker_connections = 1000

# 保持活跃连接时间
keepalive = 5

# 日志级别
loglevel = 'info'

# 预加载应用
preload_app = True

# 启用静默模式，不输出访问日志
accesslog = None

# 错误日志
errorlog = '-'  # 输出到stderr

# 进程名称
proc_name = 'xiyouji-qa'

# 绑定地址和端口
bind = '0.0.0.0:$PORT'  # 使用环境变量中的PORT

# 用户自定义设置
def on_starting(server):
    server.log.info("Gunicorn服务器正在启动，超时设置为: %s秒", timeout) 