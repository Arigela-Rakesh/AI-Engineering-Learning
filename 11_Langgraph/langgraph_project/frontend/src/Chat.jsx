
import { useState } from "react";
import "./Chat.css";

const API_URL = "http://127.0.0.1:8000/chat";

function Chat() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Keep the same thread ID throughout this chat session.
  const [threadId] = useState(
    () => `react-${Date.now()}`
  );

  async function handleSubmit(event) {
    event.preventDefault();

    const question = message.trim();

    if (!question || loading) {
      return;
    }

    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        content: question,
      },
    ]);

    setMessage("");
    setLoading(true);
    setError("");

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: question,
          thread_id: threadId,
        }),
      });

      if (!response.ok) {
        throw new Error(
          `API request failed: ${response.status}`
        );
      }

      const data = await response.json();

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources ?? [],
        },
      ]);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chat-container">
      <h2>Employee Handbook Assistant</h2>

      <div className="chat-messages">
        {messages.map((item, index) => (
          <div
            key={index}
            className={`message ${item.role}`}
          >
            <strong>
              {item.role === "user"
                ? "You"
                : "AI Assistant"}
            </strong>

            <p>{item.content}</p>

            {item.role === "assistant" && (
              <div className="sources">
                <h4>
                  Sources ({item.sources.length})
                </h4>

                {item.sources.length === 0 ? (
                  <p>No handbook sources were retrieved.</p>
                ) : (
                  item.sources.map((source) => (
                    <details
                      key={source.chunk_index}
                      className="source-card"
                    >
                      <summary>
                        Chunk {source.chunk_index}
                        {" — "}
                        Similarity:{" "}
                        {Number(source.score).toFixed(4)}
                      </summary>

                      <p>{source.excerpt}</p>
                    </details>
                  ))
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {loading && <p>Assistant is thinking...</p>}

      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}

      <form onSubmit={handleSubmit}>
        <input
          value={message}
          onChange={(event) =>
            setMessage(event.target.value)
          }
          placeholder="Ask about company policies..."
          disabled={loading}
          aria-label="Your question"
        />

        <button
          type="submit"
          disabled={loading || !message.trim()}
        >
          Send
        </button>
      </form>
    </div>
  );
}

export default Chat;
