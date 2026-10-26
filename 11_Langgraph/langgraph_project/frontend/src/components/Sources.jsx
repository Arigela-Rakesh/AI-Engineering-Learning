
import { useState } from "react";

function Sources({ sources = [] }) {
  const [expandedSources, setExpandedSources] = useState({});

  if (!sources || sources.length === 0) {
    return null;
  }

  function toggleSource(index) {
    setExpandedSources((previous) => ({
      ...previous,
      [index]: !previous[index],
    }));
  }

  return (
    <div className="sources-container">

      <h4 className="sources-heading">
        Sources ({sources.length})
      </h4>

      {sources.map((source, index) => (

        <div
          className="source-item"
          key={`${source.chunk_index}-${index}`}
        >

          <button
            className="source-toggle"
            onClick={() => toggleSource(index)}
            aria-expanded={Boolean(expandedSources[index])}
          >

            <span className="source-title">
              Handbook — Chunk {source.chunk_index}
            </span>

            <span className="source-score">
              Score: {Number(source.score).toFixed(3)}
            </span>

            <span className="source-icon">
              {expandedSources[index] ? "−" : "+"}
            </span>

          </button>

          {expandedSources[index] && (

            <div className="source-excerpt">
              {source.excerpt}
            </div>

          )}

        </div>

      ))}

    </div>
  );
}

export default Sources;
