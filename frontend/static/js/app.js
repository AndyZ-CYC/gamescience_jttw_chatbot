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
    function updateStatus(status) {
        if (status === 'thinking') {
            statusIndicator.innerHTML = '<i class="bi bi-hourglass-split"></i> 正在思考...';
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
        
        try {
            updateStatus('thinking');
            
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

            let data;
            try {
                // 尝试解析JSON响应
                data = await response.json();
            } catch (parseError) {
                // 如果解析失败，获取原始文本
                const textResponse = await response.text();
                console.error('JSON解析失败:', parseError);
                console.log('收到的非JSON响应:', textResponse.substring(0, 200) + '...');
                throw new Error('服务器未返回有效JSON格式。这通常是由于请求超时导致。请尝试使用GPT-4o模型或简化您的问题。');
            }
            
            if (!response.ok) {
                // 处理不同状态码的错误
                if (response.status === 408) {
                    throw new Error(data.message || '请求超时。请尝试使用GPT-4o模型或简化您的问题。');
                } else {
                    throw new Error(data.message || data.error || '请求失败');
                }
            }
            
            updateStatus('');
            addMessage(data.answer, 'assistant');
            
        } catch (error) {
            console.error('请求出错:', error);
            updateStatus('error');
            
            // 处理不同类型的错误，提供友好的错误消息
            let errorMessage = error.message;
            if (error.message.includes('timeout') || error.message.includes('time') && error.message.includes('out')) {
                errorMessage = '请求处理超时。请尝试使用GPT-4o模型或简化您的问题。';
            } else if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
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