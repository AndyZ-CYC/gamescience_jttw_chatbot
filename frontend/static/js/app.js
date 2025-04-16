// 初始化marked库配置
marked.use({
    gfm: true,
    breaks: true,
    pedantic: false,
    smartLists: true
});

document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const modelSelect = document.getElementById('model-select');
    const statusIndicator = document.getElementById('status-indicator');

    // 确保系统已初始化
    async function ensureInitialized() {
        try {
            updateStatus('thinking');
            addMessage('正在初始化系统，请稍候...', 'system');
            
            await fetch('/init', {
                method: 'GET'
            });
            
            updateStatus('');
            addMessage('系统初始化完成，请输入您的问题！', 'system');
        } catch (error) {
            console.error('初始化失败:', error);
            updateStatus('error');
            addMessage('系统初始化失败，请刷新页面重试。', 'system');
        }
    }

    // 初始化检查
    ensureInitialized();

    // 自动滚动到底部
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // 添加消息到聊天界面
    function addMessage(content, role) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        if (role === 'assistant') {
            // 使用Markdown渲染助手回复
            const mdContent = document.createElement('div');
            mdContent.className = 'markdown-body';
            mdContent.innerHTML = marked.parse(content);
            messageContent.appendChild(mdContent);
        } else {
            messageContent.textContent = content;
        }

        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    // 更新状态指示器
    function updateStatus(status, time) {
        if (status === 'thinking') {
            let text = '<i class="bi bi-hourglass-split"></i> 正在思考...';
            if (time) {
                text = `<i class="bi bi-hourglass-split"></i> 正在思考... (${time}秒)`;
            }
            statusIndicator.innerHTML = text;
            statusIndicator.classList.add('thinking');
        } else if (status === 'error') {
            statusIndicator.innerHTML = '<i class="bi bi-exclamation-circle"></i> 出错了';
            statusIndicator.classList.remove('thinking');
        } else {
            statusIndicator.innerHTML = '';
            statusIndicator.classList.remove('thinking');
        }
    }

    // 发送消息到服务器
    async function sendMessage(message) {
        const model = modelSelect.value;
        
        // 添加计时器变量
        let processingTimer;
        let processingTime = 0;
        
        try {
            updateStatus('thinking');
            
            // 添加定时器，每5秒更新一次状态
            processingTimer = setInterval(() => {
                processingTime += 5;
                updateStatus('thinking', processingTime);
                
                // 非GPT-4o模型时，处理时间超过60秒添加提示
                if (processingTime >= 60 && model !== 'gpt-4o' && !document.getElementById('long-wait-notice')) {
                    const noticeDiv = document.createElement('div');
                    noticeDiv.id = 'long-wait-notice';
                    noticeDiv.className = 'message system';
                    noticeDiv.innerHTML = `<div class="message-content">
                        <p>您选择的模型处理时间较长，请耐心等待。如果长时间无响应，可以尝试刷新页面并使用GPT-4o模型。</p>
                    </div>`;
                    chatMessages.appendChild(noticeDiv);
                    scrollToBottom();
                }
            }, 5000);
            
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: message,
                    model: model
                })
            });

            // 请求完成后清除定时器
            clearInterval(processingTimer);

            // 检查响应状态并先判断是否有错误
            if (!response.ok) {
                // 尝试以JSON格式获取错误详情
                try {
                    const errorData = await response.json();
                    if (response.status === 408) {
                        throw new Error(errorData.message || '请求超时。请尝试使用GPT-4o模型或简化您的问题。');
                    } else {
                        throw new Error(errorData.message || errorData.error || `服务器错误 (${response.status})`);
                    }
                } catch (jsonError) {
                    // 如果无法解析JSON，使用状态文本
                    throw new Error(`服务器返回错误: ${response.status} ${response.statusText}`);
                }
            }

            // 只有在响应成功时才尝试解析JSON
            try {
                const data = await response.json();
                updateStatus('');
                addMessage(data.answer, 'assistant');
            } catch (parseError) {
                console.error('JSON解析失败:', parseError);
                throw new Error('服务器返回了无效的数据格式。请尝试刷新页面或使用GPT-4o模型。');
            }
            
        } catch (error) {
            console.error('请求出错:', error);
            // 出错时也要清除定时器
            if (processingTimer) {
                clearInterval(processingTimer);
            }
            updateStatus('error');
            
            // 处理不同类型的错误，提供友好的错误消息
            let errorMessage = error.message;
            if (errorMessage.includes('timeout') || (errorMessage.includes('time') && errorMessage.includes('out'))) {
                errorMessage = '请求处理超时。请尝试使用GPT-4o模型或简化您的问题。';
            } else if (errorMessage.includes('Failed to fetch') || errorMessage.includes('NetworkError')) {
                errorMessage = '网络连接错误。请检查您的网络连接并重试。';
            }
            
            addMessage(`请求出错: ${errorMessage}`, 'system');
            
            // 如果是非GPT-4o模型，自动建议切换
            if (model !== 'gpt-4o') {
                setTimeout(() => {
                    addMessage('系统建议：尝试切换到GPT-4o模型可能会解决此问题。', 'system');
                    modelSelect.value = 'gpt-4o';
                }, 1000);
            }
        }
    }

    // 表单提交处理
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const message = chatInput.value.trim();
        if (!message) return;
        
        addMessage(message, 'user');
        chatInput.value = '';
        
        sendMessage(message);
    });

    // 处理Enter键提交
    chatInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });
}); 