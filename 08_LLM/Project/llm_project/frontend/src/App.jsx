
import { useState } from "react";
import {
  BrainCircuit,
  Sparkles,
  MessageSquareText,
  HeartPulse,
  LoaderCircle,
  Copy,
  Check,
  Server,
} from "lucide-react";
import "./App.css";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

async function postJson(path, payload) {
  let response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error(
      "Could not connect to the API. Make sure FastAPI is running at " +
        API_BASE_URL
    );
  }

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail = data.detail;
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg).join(", ")
      : typeof detail === "string"
        ? detail
        : `Request failed with HTTP ${response.status}`;

    throw new Error(message);
  }

  return data;
}

function App() {
  const [activeTab, setActiveTab] = useState("sentiment");

  const [sentimentText, setSentimentText] = useState("");
  const [sentimentResult, setSentimentResult] = useState(null);
  const [sentimentError, setSentimentError] = useState("");
  const [sentimentLoading, setSentimentLoading] = useState(false);

  const [prompt, setPrompt] = useState("");
  const [maxNewTokens, setMaxNewTokens] = useState(80);
  const [temperature, setTemperature] = useState(0.8);
  const [generationResult, setGenerationResult] = useState(null);
  const [generationError, setGenerationError] = useState("");
  const [generationLoading, setGenerationLoading] = useState(false);

  const [copied, setCopied] = useState(false);

  async function handleSentimentSubmit(event) {
    event.preventDefault();
    setSentimentError("");
    setSentimentResult(null);

    if (!sentimentText.trim()) {
      setSentimentError("Please enter some text to analyze.");
      return;
    }

    setSentimentLoading(true);

    try {
      const result = await postJson("/sentiment", {
        text: sentimentText.trim(),
      });
      setSentimentResult(result);
    } catch (error) {
      setSentimentError(error.message);
    } finally {
      setSentimentLoading(false);
    }
  }

  async function handleGenerationSubmit(event) {
    event.preventDefault();
    setGenerationError("");
    setGenerationResult(null);
    setCopied(false);

    if (!prompt.trim()) {
      setGenerationError("Please enter a prompt.");
      return;
    }

    setGenerationLoading(true);

    try {
      const result = await postJson("/generate", {
        prompt: prompt.trim(),
        max_new_tokens: Number(maxNewTokens),
        temperature: Number(temperature),
      });
      setGenerationResult(result);
    } catch (error) {
      setGenerationError(error.message);
    } finally {
      setGenerationLoading(false);
    }
  }

  async function copyGeneratedText() {
    if (!generationResult?.generated_text) return;

    try {
      await navigator.clipboard.writeText(
        generationResult.generated_text
      );
      setCopied(true);
    } catch {
      setGenerationError(
        "Could not copy the text. Please select and copy it manually."
      );
    }
  }

  const sentiment = sentimentResult?.sentiment?.toUpperCase();
  const isPositive = sentiment === "POSITIVE";

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="AI Text Assistant home">
          <span className="brand-icon">
            <BrainCircuit size={24} />
          </span>
          <span>
            <strong>NeuralDesk</strong>
            <small>AI TEXT ASSISTANT</small>
          </span>
        </a>

        <div className="api-status">
          <span className="status-dot" />
          <span>Local API</span>
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <div className="eyebrow">
            <Sparkles size={15} />
            YOUR PERSONAL AI WORKSPACE
          </div>

          <h1>
            Turn your words into
            <span> insights and ideas.</span>
          </h1>

          <p>
            Analyze sentiment and generate new text with your own
            Hugging Face models, powered by a FastAPI backend.
          </p>
        </section>

        <section className="workspace">
          <div className="tabs" role="tablist" aria-label="AI tools">
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === "sentiment"}
              className={`tab ${
                activeTab === "sentiment" ? "active" : ""
              }`}
              onClick={() => setActiveTab("sentiment")}
            >
              <HeartPulse size={18} />
              Sentiment analyzer
            </button>

            <button
              type="button"
              role="tab"
              aria-selected={activeTab === "generation"}
              className={`tab ${
                activeTab === "generation" ? "active" : ""
              }`}
              onClick={() => setActiveTab("generation")}
            >
              <MessageSquareText size={18} />
              Text generator
            </button>
          </div>

          {activeTab === "sentiment" ? (
            <section className="tool-panel">
              <div className="panel-heading">
                <div className="panel-icon sentiment-icon">
                  <HeartPulse size={21} />
                </div>
                <div>
                  <h2>Sentiment analysis</h2>
                  <p>Discover the emotional tone of your text.</p>
                </div>
              </div>

              <form onSubmit={handleSentimentSubmit}>
                <label htmlFor="sentiment-text">Your text</label>
                <textarea
                  id="sentiment-text"
                  value={sentimentText}
                  onChange={(event) =>
                    setSentimentText(event.target.value)
                  }
                  maxLength={5000}
                  placeholder="For example: I love this product. The quality is excellent!"
                  rows={6}
                  required
                />

                <div className="field-footer">
                  <span>Enter the text you want to analyze.</span>
                  <span>{sentimentText.length}/5000</span>
                </div>

                {sentimentError && (
                  <div className="error-message" role="alert">
                    {sentimentError}
                  </div>
                )}

                <button
                  className="primary-button"
                  type="submit"
                  disabled={sentimentLoading}
                >
                  {sentimentLoading ? (
                    <>
                      <LoaderCircle className="spin" size={18} />
                      Analyzing text...
                    </>
                  ) : (
                    <>
                      <Sparkles size={18} />
                      Analyze sentiment
                    </>
                  )}
                </button>
              </form>

              {sentimentResult && (
                <div className="result-card" aria-live="polite">
                  <div className="result-header">
                    <span className="result-label">ANALYSIS RESULT</span>
                    <span
                      className={`sentiment-badge ${
                        isPositive ? "positive" : "negative"
                      }`}
                    >
                      {sentiment}
                    </span>
                  </div>

                  <h3>
                    {isPositive
                      ? "This text sounds positive."
                      : "This text sounds negative."}
                  </h3>

                  <p className="result-description">
                    Model confidence:{" "}
                    {(
                      Number(sentimentResult.confidence) * 100
                    ).toFixed(2)}
                    %
                  </p>

                  <div className="confidence-track">
                    <div
                      className={`confidence-fill ${
                        isPositive ? "positive" : "negative"
                      }`}
                      style={{
                        width: `${Math.max(
                          0,
                          Math.min(
                            100,
                            Number(sentimentResult.confidence) * 100
                          )
                        )}%`,
                      }}
                    />
                  </div>

                  <p className="result-note">
                    Confidence is the model's estimated probability
                    for its selected class, not a guarantee of correctness.
                  </p>
                </div>
              )}
            </section>
          ) : (
            <section className="tool-panel">
              <div className="panel-heading">
                <div className="panel-icon generation-icon">
                  <MessageSquareText size={21} />
                </div>
                <div>
                  <h2>AI text generation</h2>
                  <p>Give the model a prompt and generate a continuation.</p>
                </div>
              </div>

              <form onSubmit={handleGenerationSubmit}>
                <label htmlFor="generation-prompt">Your prompt</label>
                <textarea
                  id="generation-prompt"
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  maxLength={2000}
                  placeholder="For example: Artificial intelligence can help people by..."
                  rows={5}
                  required
                />

                <div className="field-footer">
                  <span>Be specific for more useful results.</span>
                  <span>{prompt.length}/2000</span>
                </div>

                <div className="settings-grid">
                  <div className="setting">
                    <div className="setting-heading">
                      <label htmlFor="max-tokens">Max new tokens</label>
                      <strong>{maxNewTokens}</strong>
                    </div>
                    <input
                      id="max-tokens"
                      type="range"
                      min="10"
                      max="200"
                      step="10"
                      value={maxNewTokens}
                      onChange={(event) =>
                        setMaxNewTokens(Number(event.target.value))
                      }
                    />
                    <small>Controls the maximum continuation length.</small>
                  </div>

                  <div className="setting">
                    <div className="setting-heading">
                      <label htmlFor="temperature">Temperature</label>
                      <strong>{Number(temperature).toFixed(2)}</strong>
                    </div>
                    <input
                      id="temperature"
                      type="range"
                      min="0.1"
                      max="2"
                      step="0.1"
                      value={temperature}
                      onChange={(event) =>
                        setTemperature(Number(event.target.value))
                      }
                    />
                    <small>Higher values generally increase variation.</small>
                  </div>
                </div>

                {generationError && (
                  <div className="error-message" role="alert">
                    {generationError}
                  </div>
                )}

                <button
                  className="primary-button"
                  type="submit"
                  disabled={generationLoading}
                >
                  {generationLoading ? (
                    <>
                      <LoaderCircle className="spin" size={18} />
                      Generating text...
                    </>
                  ) : (
                    <>
                      <Sparkles size={18} />
                      Generate text
                    </>
                  )}
                </button>
              </form>

              {generationResult && (
                <div className="result-card" aria-live="polite">
                  <div className="result-header">
                    <span className="result-label">GENERATED OUTPUT</span>
                    <button
                      type="button"
                      className="copy-button"
                      onClick={copyGeneratedText}
                    >
                      {copied ? <Check size={16} /> : <Copy size={16} />}
                      {copied ? "Copied" : "Copy text"}
                    </button>
                  </div>

                  <h3>Your generated text</h3>
                  <p className="generated-text">
                    {generationResult.generated_text ||
                      "The model returned no new text. Try a different prompt or settings."}
                  </p>

                  <div className="generation-meta">
                    <span>
                      Max tokens:{" "}
                      {generationResult.settings?.max_new_tokens}
                    </span>
                    <span>
                      Temperature:{" "}
                      {generationResult.settings?.temperature}
                    </span>
                    <span>Model: Qwen/Qwen2.5-0.5B-Instruct</span>
                  </div>
                </div>
              )}
            </section>
          )}
        </section>

        <section className="info-strip">
          <div>
            <Server size={20} />
            <span>
              <strong>FastAPI backend</strong>
              <small>Local inference API</small>
            </span>
          </div>
          <div>
            <BrainCircuit size={20} />
            <span>
              <strong>Hugging Face</strong>
              <small>Pretrained transformer models</small>
            </span>
          </div>
          <div>
            <Sparkles size={20} />
            <span>
              <strong>Two AI tools</strong>
              <small>Classification + generation</small>
            </span>
          </div>
        </section>

        <footer className="footer">
          Built with React, FastAPI, PyTorch and Hugging Face.
        </footer>
      </main>
    </div>
  );
}

export default App;
