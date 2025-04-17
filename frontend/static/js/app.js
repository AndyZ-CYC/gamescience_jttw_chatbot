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

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || '请求失败');
            }
            
            updateStatus('');
            addMessage(data.answer, 'assistant');
            
        } catch (error) {
            console.error('请求出错:', error);
            updateStatus('error');
            addMessage(`请求出错: ${error.message}`, 'system');
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