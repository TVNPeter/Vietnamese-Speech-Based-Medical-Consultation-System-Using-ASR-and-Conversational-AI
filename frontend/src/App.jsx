import { useCallback, useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Bot,
  ChevronRight,
  CircleStop,
  HeartPulse,
  Mic,
  Paperclip,
  SendHorizontal,
  ShieldAlert,
  Sparkles,
  Trash2,
  UserRound,
  Volume2,
  X,
} from 'lucide-react';
import './index.css';

const API_BASE = import.meta.env.VITE_API_BASE || '/api';
const STORAGE_KEY = 'medical_chat_history_v2';

const WELCOME_MESSAGE = {
  id: 'welcome',
  role: 'assistant',
  content:
    'Chào bạn, mình là trợ lý y tế AI. Bạn có thể hỏi về triệu chứng, thuốc hoặc tương tác thuốc. Thông tin chỉ mang tính tham khảo và không thay thế khám bệnh.',
  sources: [],
};

const SUGGESTED_QUESTIONS = [
  'Tác dụng phụ của warfarin là gì?',
  'Panadol có thể dùng khi nào?',
  'Dấu hiệu nào cần đi khám ngay?',
];

function loadMessages() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
    if (Array.isArray(saved) && saved.length) {
      return saved.map((message) => ({ ...message, isStreaming: false }));
    }
  } catch {
    localStorage.removeItem(STORAGE_KEY);
  }
  return [WELCOME_MESSAGE];
}

function stripMarkdown(text) {
  return text
    .replace(/```[\s\S]*?```/g, '')
    .replace(/[#*_>`]/g, '')
    .replace(/\[(.*?)\]\([^)]*\)/g, '$1')
    .replace(/\[(?:\d+(?:\s*[,–-]\s*\d+)*)\]/g, '')
    .trim();
}

function Sources({ sources }) {
  if (!sources?.length) return null;

  return (
    <details className="sources" open={false}>
      <summary>Nguồn tham khảo ({sources.length})</summary>
      <ol>
        {sources.map((source) => (
          <li key={`${source.index}-${source.title}`}>
            <strong>{source.title || 'Tài liệu y khoa'}</strong>
            {source.url && (
              <a href={source.url} target="_blank" rel="noreferrer">
                Mở nguồn
              </a>
            )}
            {source.content && <span>{source.content}</span>}
          </li>
        ))}
      </ol>
    </details>
  );
}

function QuickPrompts({ onSelect }) {
  return (
    <section className="quick-prompts" aria-label="Câu hỏi gợi ý">
      <div className="quick-prompts-heading">
        <span><Sparkles size={16} /></span>
        <div>
          <p>Bắt đầu cuộc trò chuyện</p>
          <h2>Hỏi nhanh về thuốc hoặc triệu chứng</h2>
        </div>
      </div>
      <div className="quick-prompts-list">
        {SUGGESTED_QUESTIONS.map((question) => (
          <button key={question} onClick={() => onSelect(question)}>
            <span>{question}</span>
            <ChevronRight size={17} />
          </button>
        ))}
      </div>
    </section>
  );
}

function App() {
  const [messages, setMessages] = useState(loadMessages);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [playingId, setPlayingId] = useState(null);
  const [notice, setNotice] = useState('');
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const abortControllerRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioRef = useRef(null);
  const audioFileInputRef = useRef(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading, scrollToBottom]);

  useEffect(() => {
    const savedMessages = messages.map((message) => {
      const savedMessage = { ...message };
      delete savedMessage.isStreaming;
      return savedMessage;
    });
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(savedMessages),
    );
  }, [messages]);

  useEffect(
    () => () => {
      abortControllerRef.current?.abort();
      audioRef.current?.pause();
    },
    [],
  );

  const resizeTextarea = () => {
    if (!textareaRef.current) return;
    textareaRef.current.style.height = 'auto';
    textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 140)}px`;
  };

  const updateAssistant = (id, update) => {
    setMessages((previous) =>
      previous.map((message) =>
        message.id === id ? { ...message, ...update } : message,
      ),
    );
  };

  const handleSend = async () => {
    const question = input.trim();
    if (!question || isLoading) return;

    const userId = crypto.randomUUID();
    const assistantId = crypto.randomUUID();
    const controller = new AbortController();
    abortControllerRef.current = controller;
    setInput('');
    setNotice('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    setMessages((previous) => [
      ...previous,
      { id: userId, role: 'user', content: question },
      {
        id: assistantId,
        role: 'assistant',
        content: '',
        sources: [],
        isStreaming: true,
      },
    ]);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: question }),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        throw new Error(`Máy chủ trả về mã ${response.status}.`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let answer = '';

      const processEvent = (event) => {
        const payload = event
          .split('\n')
          .filter((line) => line.startsWith('data: '))
          .map((line) => line.slice(6))
          .join('');
        if (!payload) return;

        const data = JSON.parse(payload);
        if (data.type === 'metadata') {
          updateAssistant(assistantId, { sources: data.data?.sources || [] });
        } else if (data.type === 'token') {
          answer += data.content || '';
          updateAssistant(assistantId, { content: answer });
        } else if (data.type === 'error') {
          throw new Error(data.content || 'Không thể tạo câu trả lời.');
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';
        events.forEach(processEvent);
      }
      if (buffer.trim()) processEvent(buffer);
      updateAssistant(assistantId, {
        content: answer || 'Không nhận được nội dung trả lời từ máy chủ.',
        isStreaming: false,
      });
    } catch (error) {
      if (error.name === 'AbortError') {
        updateAssistant(assistantId, {
          content: 'Đã dừng tạo câu trả lời.',
          isStreaming: false,
        });
      } else {
        updateAssistant(assistantId, {
          content: `**Không thể kết nối:** ${error.message}`,
          isStreaming: false,
        });
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  const stopStreaming = () => abortControllerRef.current?.abort();

  const handleClearChat = () => {
    if (!window.confirm('Xóa toàn bộ lịch sử trò chuyện trên trình duyệt này?')) {
      return;
    }
    abortControllerRef.current?.abort();
    localStorage.removeItem(STORAGE_KEY);
    setMessages([WELCOME_MESSAGE]);
    setInput('');
  };

  const stopAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      URL.revokeObjectURL(audioRef.current.src);
      audioRef.current = null;
    }
    setPlayingId(null);
  };

  const handleSpeak = async (message) => {
    if (playingId === message.id) {
      stopAudio();
      return;
    }
    stopAudio();
    setPlayingId(message.id);

    try {
      const response = await fetch(`${API_BASE}/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: stripMarkdown(message.content) }),
      });
      if (!response.ok) throw new Error(`Máy chủ trả về mã ${response.status}.`);

      const audioUrl = URL.createObjectURL(await response.blob());
      const audio = new Audio(audioUrl);
      audioRef.current = audio;
      audio.onended = stopAudio;
      await audio.play();
    } catch (error) {
      setNotice(`Không thể phát âm thanh: ${error.message}`);
      setPlayingId(null);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  const transcribeAudio = async (audio, filename) => {
    if (!audio || isLoading || isTranscribing) return;

    const formData = new FormData();
    formData.append('file', audio, filename || 'recording.wav');
    setIsTranscribing(true);
    setNotice(`Đang nhận dạng ${filename || 'file âm thanh'}…`);
    try {
      const response = await fetch(`${API_BASE}/asr`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(`Máy chủ trả về mã ${response.status}.`);
      const result = await response.json();
      setInput((previous) => `${previous} ${result.text || ''}`.trim());
      setNotice('');
      requestAnimationFrame(resizeTextarea);
    } catch (error) {
      setNotice(`Không thể nhận dạng giọng nói: ${error.message}`);
    } finally {
      setIsTranscribing(false);
    }
  };

  const handleAudioFile = (event) => {
    const [audio] = event.target.files || [];
    event.target.value = '';
    if (audio) transcribeAudio(audio, audio.name);
  };

  const startRecording = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setNotice('Trình duyệt này không hỗ trợ ghi âm.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = recorder;
      audioChunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size) audioChunksRef.current.push(event.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        const audio = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await transcribeAudio(audio, 'recording.webm');
      };
      recorder.start();
      setIsRecording(true);
      setNotice('Đang ghi âm… bấm mic lần nữa để dừng.');
    } catch (error) {
      setNotice(`Không thể dùng microphone: ${error.message}`);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  const selectSuggestedQuestion = (question) => {
    setInput(question);
    requestAnimationFrame(resizeTextarea);
    textareaRef.current?.focus();
  };

  return (
    <main className="app-shell">
      <section className="chat-card" aria-label="Trợ lý y tế AI">
        <header className="app-header">
          <div className="brand">
            <span className="brand-icon"><HeartPulse size={23} /></span>
            <div>
              <p>Hệ thống tư vấn y tế</p>
              <h1>Trợ lý Y tế AI</h1>
            </div>
          </div>
          <div className="header-actions">
            <span className="status"><i /> Sẵn sàng tư vấn</span>
            <button className="icon-button" onClick={handleClearChat} title="Xóa lịch sử chat" aria-label="Xóa lịch sử chat">
              <Trash2 size={18} />
            </button>
          </div>
        </header>

        <div className="medical-note">
          <ShieldAlert size={17} aria-hidden="true" />
          <span>Không dùng cho tình huống cấp cứu. Nếu có dấu hiệu nguy hiểm, hãy gọi cấp cứu hoặc đến cơ sở y tế gần nhất.</span>
        </div>

        <section className={`conversation ${messages.length === 1 ? 'is-empty' : ''}`} aria-live="polite">
          {messages.map((message) => (
            <article key={message.id} className={`message ${message.role}`}>
              <div className="avatar" aria-hidden="true">
                {message.role === 'assistant' ? <Bot size={18} /> : <UserRound size={18} />}
              </div>
              <div className="message-body">
                <div className="message-label">
                  {message.role === 'assistant' ? 'Trợ lý Y tế AI' : 'Bạn'}
                </div>
                <div className="message-content">
                  {message.role === 'assistant' ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                  ) : (
                    message.content
                  )}
                  {message.isStreaming && <span className="stream-cursor" aria-label="Đang trả lời" />}
                </div>
                {message.role === 'assistant' && !message.isStreaming && message.content && (
                  <div className="message-tools">
                    <button className="speak-button" onClick={() => handleSpeak(message)}>
                      {playingId === message.id ? <X size={15} /> : <Volume2 size={15} />}
                      {playingId === message.id ? 'Dừng đọc' : 'Nghe trả lời'}
                    </button>
                    <Sources sources={message.sources} />
                  </div>
                )}
              </div>
            </article>
          ))}
          {messages.length === 1 && !isLoading && (
            <QuickPrompts onSelect={selectSuggestedQuestion} />
          )}
          {isLoading && <div className="typing"><span /><span /><span /> Đang tìm tài liệu và soạn câu trả lời…</div>}
          <div ref={messagesEndRef} />
        </section>

        <footer className="composer-area">
          {notice && <p className="notice">{notice}</p>}
          <div className="composer">
            <input
              ref={audioFileInputRef}
              className="audio-file-input"
              type="file"
              accept="audio/wav,audio/x-wav,audio/webm,.wav,.webm"
              onChange={handleAudioFile}
              aria-label="Chọn file âm thanh"
            />
            <button
              className="upload-button"
              onClick={() => audioFileInputRef.current?.click()}
              disabled={isLoading || isTranscribing}
              title="Chọn file WAV hoặc WebM để nhận dạng"
              aria-label="Chọn file âm thanh để nhận dạng"
            >
              <Paperclip size={19} />
            </button>
            <button
              className={`mic-button ${isRecording ? 'recording' : ''}`}
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isLoading || isTranscribing}
              title={isRecording ? 'Dừng ghi âm' : 'Ghi âm câu hỏi'}
              aria-label={isRecording ? 'Dừng ghi âm' : 'Ghi âm câu hỏi'}
            >
              <Mic size={20} />
            </button>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(event) => {
                setInput(event.target.value);
                resizeTextarea();
              }}
              onKeyDown={handleKeyDown}
              placeholder="Nhập câu hỏi y tế của bạn…"
              rows={1}
              disabled={isLoading}
              aria-label="Câu hỏi y tế"
            />
            {isLoading ? (
              <button className="stop-button" onClick={stopStreaming} title="Dừng trả lời" aria-label="Dừng trả lời">
                <CircleStop size={20} />
              </button>
            ) : (
              <button className="send-button" onClick={handleSend} disabled={!input.trim() || isTranscribing} title="Gửi câu hỏi" aria-label="Gửi câu hỏi">
                <SendHorizontal size={20} />
              </button>
            )}
          </div>
          <p className="composer-hint">Kẹp giấy để chọn WAV/WebM · Enter để gửi · Câu trả lời cần được bác sĩ xác nhận khi có quyết định điều trị.</p>
        </footer>
      </section>
    </main>
  );
}

export default App;
