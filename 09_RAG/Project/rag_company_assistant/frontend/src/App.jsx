
import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [apiStatus, setApiStatus] = useState("checking");
  const [apiMessage, setApiMessage] = useState("");
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(2);
  const [minScore, setMinScore] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Check whether the FastAPI backend is available.
  useEffect(() => {
    let active = true;

    async function checkHealth() {
      try {
        const response = await fetch(`${API_URL}/health`);

        if (!response.ok) {
          throw new Error("Health check failed");
        }

        const data = await response.json();

        if (active) {
          if (data.pipeline_ready) {
            setApiStatus("online");
            setApiMessage("Backend ready");
          } else {
            setApiStatus("starting");
            setApiMessage("Backend is initializing");
          }
        }
      } catch {
        if (active) {
          setApiStatus("offline");
          setApiMessage("Backend unavailable");
        }
      }
    }

    checkHealth();

    return () => {
      active = false;
    };
  }, []);

  // Submit a question to the RAG API.
  async function handleSubmit(event) {
    event.preventDefault();

    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }

    if (
      minScore.trim() !== "" &&
      (!Number.isFinite(Number(minScore)) ||
        Number(minScore) < -1 ||
        Number(minScore) > 1)
    ) {
      setError("Minimum similarity must be between -1 and 1.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question.trim(),
          top_k: Number(topK),
          min_score:
            minScore.trim() === "" ? null : Number(minScore),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Unable to get an answer."
        );
      }

      setResult(data);
      setApiStatus("online");
      setApiMessage("Backend ready");
    } catch (err) {
      if (err.message === "Failed to fetch") {
        setError(
          "Cannot connect to the API. Check that FastAPI is running on port 8000 and CORS is configured."
        );
        setApiStatus("offline");
        setApiMessage("Backend unavailable");
      } else {
        setError(err.message || "An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">R</div>

          <div>
            <h1>RAG Assistant</h1>
            <p>Company knowledge, made simple</p>
          </div>
        </div>

        <span className={`status status-${apiStatus}`}>
          <span className="status-dot" />

          {apiStatus === "checking"
            ? "Checking API..."
            : apiStatus === "online"
              ? "API Online"
              : apiStatus === "starting"
                ? "API Starting"
                : "API Offline"}
        </span>
      </header>

      <p className="api-status-message" role="status">
        {apiMessage}
      </p>

      <section className="hero">
        <span className="eyebrow">
          COMPANY KNOWLEDGE ASSISTANT
        </span>

        <h2>Ask your documents.</h2>

        <p>
          Ask a question about company policies and get an answer
          grounded in your indexed documents.
        </p>
      </section>

      <section className="workspace">
        <form className="question-card" onSubmit={handleSubmit}>
          <label htmlFor="question">
            What would you like to know?
          </label>

          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="e.g. What is the internet reimbursement limit?"
            rows={3}
            maxLength={1000}
            disabled={loading}
          />

          <div className="settings">
            <div className="setting">
              <label htmlFor="topK">Retrieved chunks</label>

              <select
                id="topK"
                value={topK}
                onChange={(event) => setTopK(event.target.value)}
                disabled={loading}
              >
                <option value={1}>1 chunk</option>
                <option value={2}>2 chunks</option>
                <option value={3}>3 chunks</option>
                <option value={4}>4 chunks</option>
                <option value={5}>5 chunks</option>
              </select>
            </div>

            <div className="setting">
              <label htmlFor="minScore">
                Minimum similarity (optional)
              </label>

              <input
                id="minScore"
                type="number"
                min="-1"
                max="1"
                step="0.05"
                value={minScore}
                onChange={(event) => setMinScore(event.target.value)}
                placeholder="e.g. 0.30"
                disabled={loading}
              />
            </div>
          </div>

          <button type="submit" disabled={loading}>
            {loading ? "Searching documents..." : "Ask question"}
            {!loading && <span aria-hidden="true"> →</span>}
          </button>

          {error && (
            <p className="error-message" role="alert">
              {error}
            </p>
          )}
        </form>

        {loading && (
          <div className="message-card" role="status">
            <span className="loader" />
            Searching relevant chunks and generating an answer...
          </div>
        )}

        {result && !loading && (
          <section className="results" aria-live="polite">
            <div className="answer-card">
              <span className="section-label">YOUR QUESTION</span>

              <h3>{result.question}</h3>

              <div className="answer-divider" />

              <span className="section-label">ANSWER</span>

              <p className="answer-text">{result.answer}</p>
            </div>

            <div className="sources-card">
              <h3>Source citations</h3>

              {result.sources?.length ? (
                <ul className="source-list">
                  {result.sources.map((source, index) => (
                    <li
                      key={`${source.source}-${source.chunk_id}-${index}`}
                    >
                      <span className="source-icon">↗</span>

                      <div>
                        <strong>{source.source}</strong>

                        <p>
                          {source.page == null
                            ? "Page not available"
                            : `Page ${source.page}`}
                          {" · "}
                          Chunk {source.chunk_id}
                        </p>
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="muted">
                  No relevant sources were retrieved.
                </p>
              )}
            </div>

            <details className="chunks-card">
              <summary>
                Retrieved chunks
                <span>
                  {result.retrieved_chunks?.length ?? 0} results
                </span>
              </summary>

              {result.retrieved_chunks?.length ? (
                result.retrieved_chunks.map((chunk, index) => (
                  <article
                    className="chunk"
                    key={`${chunk.source}-${chunk.chunk_id}-${index}`}
                  >
                    <div className="chunk-heading">
                      <strong>
                        {chunk.source ?? "Document"}
                        {chunk.chunk_id != null
                          ? ` · Chunk ${chunk.chunk_id}`
                          : ""}
                      </strong>

                      {typeof chunk.score === "number" && (
                        <span>
                          Score: {chunk.score.toFixed(3)}
                        </span>
                      )}
                    </div>

                    <p>{chunk.text}</p>
                  </article>
                ))
              ) : (
                <p className="muted">No chunks available.</p>
              )}
            </details>
          </section>
        )}

        <p className="disclaimer">
          Answers are generated from indexed company documents.
          Always verify important policy decisions against official sources.
        </p>
      </section>
    </main>
  );
}

export default App;
