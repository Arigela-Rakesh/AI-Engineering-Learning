import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import Sources from "./components/Sources";

import "./App.css";

const API_URL = "http://127.0.0.1:8000/chat";

function createThreadId() {
  return crypto.randomUUID();
}

function App() {
  const messagesEndRef = useRef(null);

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [threadId, setThreadId] = useState(createThreadId);
  const [error, setError] = useState("");

  // Scroll to the latest message.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  // Send a message to FastAPI.
  async function handleSend(messageText = input) {
    const message = messageText.trim();

    if (!message || loading) {
      return;
    }

    setMessages((previousMessages) => [
      ...previousMessages,
      {
        role: "user",
        content: message,
      },
    ]);

    setInput("");
    setLoading(true);
    setError("");

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message,
          thread_id: threadId,
        }),
      });

      if (!response.ok) {
        let errorMessage =
          `Backend request failed: ${response.status}`;

        try {
          const errorData = await response.json();

          if (errorData.detail) {
            errorMessage =
              typeof errorData.detail === "string"
                ? errorData.detail
                : JSON.stringify(errorData.detail);
          }
        } catch {
          // Keep the default error message.
        }

        throw new Error(errorMessage);
      }

      const data = await response.json();

      if (typeof data.answer !== "string") {
        throw new Error(
          "The backend returned an invalid answer."
        );
      }

      const sources = Array.isArray(data.sources)
        ? data.sources.filter(
            (source) =>
              source &&
              typeof source.chunk_index === "number" &&
              typeof source.score === "number" &&
              typeof source.excerpt === "string"
          )
        : [];

      setMessages((previousMessages) => [
        ...previousMessages,
        {
          role: "assistant",
          content: data.answer,
          sources,
          toolCall: false,
        },
      ]);
    } catch (requestError) {
      console.error("Chat API error:", requestError);

      setError(
        requestError instanceof Error
          ? requestError.message
          : "Unable to connect to the AI assistant."
      );
    } finally {
      setLoading(false);
    }
  }

  // Enter sends; Shift+Enter inserts a newline.
  function handleKeyDown(event) {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      handleSend();
    }
  }

  // Start a fresh conversation.
  function handleNewChat() {
    if (loading) {
      return;
    }

    setMessages([]);
    setInput("");
    setError("");
    setThreadId(createThreadId());
  }

  // Only display user messages and final assistant answers.
  // Never render tool-call objects as chat messages.
  const visibleMessages = messages.filter(
    (message) =>
      (message.role === "user" ||
        message.role === "assistant") &&
      !message.toolCall &&
      typeof message.content === "string" &&
      message.content.trim() !== ""
  );

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">AI</div>
          <h2>LangGraph AI</h2>
        </div>

        <button
          className="new-chat-button"
          onClick={handleNewChat}
          disabled={loading}
        >
          + New chat
        </button>

        <div className="sidebar-info">
          <p>Powered by</p>
          <strong>LangGraph + Ollama</strong>
          <span>Persistent AI conversations</span>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="chat-area">
        {/* Header */}
        <header className="chat-header">
          <div>
            <h1>AI Assistant</h1>
            <p>
              Calculations • General Knowledge • Company Handbook
            </p>
          </div>

          <div className="status">
            <span className="status-dot"></span>
            Local AI
          </div>
        </header>

        {/* Messages */}
        <section
          className="messages"
          aria-live="polite"
          aria-busy={loading}
        >
          {/* Welcome Screen */}
          {visibleMessages.length === 0 && (
            <div className="welcome">
              <div className="welcome-icon">✦</div>

              <h2>How can I help you today?</h2>

              <p>
                Your AI assistant for mathematical calculations,
                general knowledge, and company handbook questions.
              </p>

              <div className="suggestions">
                <div className="suggestion-card">
                  <span>🧮</span>
                  <p>Mathematical Calculations</p>
                </div>

                <div className="suggestion-card">
                  <span>🌍</span>
                  <p>General Knowledge</p>
                </div>

                <div className="suggestion-card">
                  <span>📘</span>
                  <p>Company Handbook & Policies</p>
                </div>
              </div>
            </div>
          )}

          {/* Chat Messages */}
          {visibleMessages.map((message, index) => (
            <div
              key={`${threadId}-${index}`}
              className={`message-row ${message.role}`}
            >
              <div className="message-avatar">
                {message.role === "user" ? "You" : "AI"}
              </div>

              <div className="message-content">
                <strong>
                  {message.role === "user"
                    ? "You"
                    : "Assistant"}
                </strong>

                <div className="message-markdown">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {message.content}
                  </ReactMarkdown>
                </div>

                {message.role === "assistant" && (
                  <Sources sources={message.sources ?? []} />
                )}
              </div>
            </div>
          ))}

          {/* Loading Indicator */}
          {loading && (
            <div className="message-row assistant">
              <div className="message-avatar">AI</div>

              <div className="message-content">
                <strong>Assistant</strong>
                <p className="typing">Thinking...</p>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </section>

        {/* Error Message */}
        {error && (
          <div className="error-message" role="alert">
            {error}
          </div>
        )}

        {/* Input Area */}
        <div className="input-area">
          <div className="input-box">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a calculation, general question, or handbook policy..."
              rows={1}
              disabled={loading}
              aria-label="Your question"
            />

            <button
              className="send-button"
              onClick={() => handleSend()}
              disabled={!input.trim() || loading}
              aria-label="Send message"
            >
              ↑
            </button>
          </div>

          <p className="input-footer">
            LangGraph AI Assistant • Press Enter to send
          </p>
        </div>
      </main>
    </div>
  );
}

export default App;
