import { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Send, Trash2, Mic, Volume2, Square } from 'lucide-react';
import './index.css';

const API_BASE = '/api';

const WELCOME_MESSAGE = {
  id: 1,
  role: 'bot',
  content:
    'Chào bạn, mình là Trợ lý Y tế AI. Bạn cần tư vấn thông tin gì về sức khỏe hay thuốc men hôm nay?',
};

function App() {
  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem('chat_messages');
    if (saved) {
      const parsed = JSON.parse(saved);
      return parsed.map((msg) => ({ ...msg, _isStreaming: false }));
    }
    return [WELCOME_MESSAGE];
  });
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [playingId, setPlayingId] = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const abortControllerRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioRef = useRef(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading, isStreaming, scrollToBottom]);

  useEffect(() => {
    localStorage.setItem('chat_messages', JSON.stringify(messages));
  }, [messages]);

  const handleInput = (e) => {
    setInput(e.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = '24px';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  };

  const handleClearChat = () => {
    if (
      window.confirm('Bạn có chắc chắn muốn xóa toàn bộ lịch sử trò chuyện?')
    ) {
      localStorage.removeItem('chat_messages');
      setMessages([WELCOME_MESSAGE]);
    }
  };

  /* ==================== SEND MESSAGE ==================== */
  const handleSend = async () => {
    if (!input.trim() || isLoading || isStreaming) return;

    const userMsg = input.trim();
    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = '24px';

    const userMsgId = Date.now();
    const botMsgId = userMsgId + 1;

    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: 'user', content: userMsg },
    ]);
    setIsLoading(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let currentContent = '';
      let buffer = '';
      let botMessageAdded = false;

      let pendingContent = null;
      let rafId = null;

      const flushContent = () => {
        if (pendingContent !== null) {
          const contentToSet = pendingContent;
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.id === botMsgId) {
              updated[updated.length - 1] = { ...last, content: contentToSet };
            }
            return updated;
          });
          pendingContent = null;
        }
        rafId = null;
      };

      const scheduleUpdate = (content) => {
        pendingContent = content;
        if (!rafId) {
          rafId = requestAnimationFrame(flushContent);
        }
      };

      const ensureBotMessage = () => {
        if (!botMessageAdded) {
          setMessages((prev) => [
            ...prev,
            { id: botMsgId, role: 'bot', content: '', _isStreaming: true },
          ]);
          setIsLoading(false);
          setIsStreaming(true);
          botMessageAdded = true;
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (const part of parts) {
          const trimmed = part.trim();
          if (!trimmed.startsWith('data: ')) continue;

          const data = JSON.parse(trimmed.slice(6));

          if (data.type === 'metadata') {
            ensureBotMessage();
          } else if (data.type === 'token') {
            ensureBotMessage();
            currentContent += data.content;
            scheduleUpdate(currentContent);
          } else if (data.type === 'error') {
            throw new Error(data.content);
          }
        }
      }

      if (rafId) {
        cancelAnimationFrame(rafId);
      }
      flushContent();

      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last && last.id === botMsgId) {
          updated[updated.length - 1] = { ...last, _isStreaming: false };
        }
        return updated;
      });
    } catch (error) {
      if (error.name === 'AbortError') return;
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: 'bot',
          content: `**Lỗi kết nối:** ${error.message}`,
        },
      ]);
    } finally {
      setIsLoading(false);
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  /* ==================== VOICE RECORDING ==================== */
  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;
    audioChunksRef.current = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        audioChunksRef.current.push(e.data);
      }
    };

    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop());
      const audioBlob = new Blob(audioChunksRef.current, {
        type: 'audio/webm',
      });
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm');

      try {
        const res = await fetch(`${API_BASE}/asr`, {
          method: 'POST',
          body: formData,
        });
        if (!res.ok) throw new Error(`ASR error: ${res.status}`);
        const result = await res.json();
        setInput((prev) => (prev ? `${prev} ${result.text}` : result.text));
      } catch (err) {
        console.error('ASR failed:', err);
        alert(`Lỗi nhận dạng giọng nói: ${err.message}`);
      }
    };

    mediaRecorder.start();
    setIsRecording(true);
  };

  const stopRecording = () => {
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state === 'recording'
    ) {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  /* ==================== TTS PLAYBACK ==================== */
  const handleSpeak = async (messageId, text) => {
    if (playingId === messageId) {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      setPlayingId(null);
      return;
    }

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }

    setPlayingId(messageId);

    try {
      const res = await fetch(`${API_BASE}/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`TTS error: ${res.status}`);

      const audioBlob = await res.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audioRef.current = audio;

      audio.onended = () => {
        setPlayingId(null);
        URL.revokeObjectURL(audioUrl);
        audioRef.current = null;
      };

      audio.play();
    } catch (err) {
      console.error('TTS failed:', err);
      setPlayingId(null);
      alert(`Lỗi phát âm thanh: ${err.message}`);
    }
  };

  /* ==================== RENDER ==================== */
  return (
    <div className="app-container">
      <header className="header">
        <h1>🏥 Trợ Lý Y Tế AI</h1>
        <button
          className="clear-btn"
          onClick={handleClearChat}
          title="Xóa lịch sử trò chuyện"
        >
          <Trash2 size={18} />
        </button>
      </header>

      <main className="chat-window">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.role}`}>
            <div className="msg-content">
              {msg.role === 'bot' ? (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {msg.content}
                </ReactMarkdown>
              ) : (
                msg.content
              )}
            </div>
            {msg.role === 'bot' && msg.content && !msg._isStreaming && (
              <button
                className={`speak-btn ${playingId === msg.id ? 'playing' : ''}`}
                onClick={() => handleSpeak(msg.id, msg.content)}
                title={playingId === msg.id ? 'Dừng phát' : 'Nghe phản hồi'}
              >
                {playingId === msg.id ? (
                  <Square size={14} />
                ) : (
                  <Volume2 size={14} />
                )}
              </button>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="typing-indicator">
            <div className="dot"></div>
            <div className="dot"></div>
            <div className="dot"></div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      <footer className="input-area">
        <div className="input-wrapper">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Hỏi về triệu chứng, thuốc, hoặc lời khuyên y tế..."
            rows={1}
            disabled={isLoading || isStreaming}
          />
          <button
            className={`mic-btn ${isRecording ? 'recording' : ''}`}
            onClick={toggleRecording}
            disabled={isLoading || isStreaming}
            title={isRecording ? 'Dừng ghi âm' : 'Ghi âm giọng nói'}
          >
            <Mic size={20} />
          </button>
          <button
            className="send-btn"
            onClick={handleSend}
            disabled={!input.trim() || isLoading || isStreaming}
          >
            <Send size={20} />
          </button>
        </div>
      </footer>
    </div>
  );
}

export default App;
