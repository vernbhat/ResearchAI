import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BookOpen,
  FileText,
  GitCompare,
  History,
  Loader2,
  MessageSquareText,
  Search,
  Sparkles,
  Upload,
} from "lucide-react";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function api(path, options) {
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "Request failed");
  }
  return response.json();
}

function App() {
  const [papers, setPapers] = useState([]);
  const [chats, setChats] = useState([]);
  const [question, setQuestion] = useState("");
  const [chatId, setChatId] = useState(null);
  const [answer, setAnswer] = useState(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [selected, setSelected] = useState([]);
  const [summary, setSummary] = useState("");
  const [comparison, setComparison] = useState([]);
  const [related, setRelated] = useState([]);
  const [relatedSearched, setRelatedSearched] = useState(false);

  const hasPapers = papers.length > 0;

  async function refresh() {
    const [paperData, chatData] = await Promise.all([
      api("/papers"),
      api("/chats"),
    ]);
    setPapers(paperData);
    setChats(chatData);
  }

  useEffect(() => {
    refresh().catch((err) => setError(err.message));
  }, []);

  async function upload(files) {
    if (!files?.length) return;
    setBusy("upload");
    setError("");
    try {
      for (const file of files) {
        const form = new FormData();
        form.append("file", file);
        await api("/papers", { method: "POST", body: form });
      }
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function ask() {
    if (!question.trim()) return;
    setBusy("ask");
    setError("");
    try {
      const data = await api("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, chat_id: chatId }),
      });
      setAnswer(data);
      setChatId(data.chat_id);
      setQuestion("");
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function loadSummary(paperId) {
    setBusy(`summary-${paperId}`);
    setError("");
    try {
      const data = await api(`/papers/${paperId}/summary`);
      setSummary(data.summary);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function compare() {
    setBusy("compare");
    setError("");
    try {
      const data = await api("/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paper_ids: selected }),
      });
      setComparison(data.comparison);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function findRelated(paperId) {
    setBusy(`related-${paperId}`);
    setError("");
    setRelatedSearched(true);
    try {
      const data = await api("/related", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paper_id: paperId }),
      });
      setRelated(data.related);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  const selectedTitles = useMemo(
    () =>
      papers
        .filter((paper) => selected.includes(paper.id))
        .map((paper) => paper.title),
    [papers, selected],
  );

  return (
    <main>
      <aside>
        <div className="brand">
          <BookOpen size={26} />
          <div>
            <h1>Research AI</h1>
            <p>Literature review workspace</p>
          </div>
        </div>

        <label className="upload">
          <Upload size={22} />
          <span>{busy === "upload" ? "Indexing PDFs..." : "Upload PDFs"}</span>
          <input
            type="file"
            accept="application/pdf"
            multiple
            onChange={(e) => upload(e.target.files)}
          />
        </label>

        <section className="panel">
          <h2>
            <FileText size={17} /> Papers
          </h2>
          {papers.map((paper) => (
            <div className="paper" key={paper.id}>
              <label>
                <input
                  type="checkbox"
                  checked={selected.includes(paper.id)}
                  onChange={(event) =>
                    setSelected((current) =>
                      event.target.checked
                        ? [...current, paper.id]
                        : current.filter((id) => id !== paper.id),
                    )
                  }
                />
                <span>{paper.title}</span>
              </label>
              <small>
                {paper.page_count} pages · {paper.chunk_count} chunks
              </small>
              <div className="row">
                <button onClick={() => loadSummary(paper.id)}>Summary</button>
                <button onClick={() => findRelated(paper.id)}>Related</button>
              </div>
            </div>
          ))}
          {!hasPapers && (
            <p className="empty">
              Upload PDFs to build your local research index.
            </p>
          )}
        </section>

        <section className="panel">
          <h2>
            <History size={17} /> Chats
          </h2>
          {chats.slice(0, 8).map((chat) => (
            <button
              className="chat-link"
              key={chat.id}
              onClick={() => setChatId(chat.id)}
            >
              {chat.title}
            </button>
          ))}
          {!chats.length && (
            <p className="empty">Saved conversations will appear here.</p>
          )}
        </section>
      </aside>

      <section className="workspace">
        <div className="topbar">
          <div>
            <h2>Ask, compare, and cite your papers</h2>
            <p>
              {selectedTitles.length
                ? `Selected: ${selectedTitles.join(", ")}`
                : "Grounded answers use only uploaded PDF chunks."}
            </p>
          </div>
          <button
            className="primary"
            disabled={selected.length < 2 || busy === "compare"}
            onClick={compare}
          >
            <GitCompare size={18} />
            Compare
          </button>
        </div>

        {error && <div className="error">{error}</div>}

        <section className="ask">
          <MessageSquareText size={22} />
          <textarea
            placeholder="Ask a question, e.g. What limitations do these papers report?"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey))
                ask();
            }}
          />
          <button
            className="primary"
            disabled={!hasPapers || busy === "ask"}
            onClick={ask}
          >
            {busy === "ask" ? (
              <Loader2 className="spin" size={18} />
            ) : (
              <Search size={18} />
            )}
            Ask
          </button>
        </section>

        <div className="grid">
          <section className="result">
            <h3>
              <Sparkles size={18} /> Answer
            </h3>
            {answer ? (
              <>
                <pre>{answer.answer}</pre>
                <div className="citations">
                  {answer.citations.map((citation) => (
                    <article key={citation.chunk_id}>
                      <strong>{citation.paper_title}</strong>
                      <span>
                        Page {citation.page_number} · score {citation.score}
                      </span>
                      <p>{citation.excerpt}</p>
                    </article>
                  ))}
                </div>
              </>
            ) : (
              <p className="empty">
                Ask a question to retrieve cited evidence.
              </p>
            )}
          </section>

          <section className="result">
            <h3>
              <FileText size={18} /> Summary
            </h3>
            {summary ? (
              <pre>{summary}</pre>
            ) : (
              <p className="empty">Choose Summary on any paper.</p>
            )}
          </section>
        </div>

        <div className="grid">
          <section className="result">
            <h3>
              <GitCompare size={18} /> Comparison
            </h3>
            {comparison.length ? (
              comparison.map((entry) => (
                <article key={entry.paper.id}>
                  <strong>{entry.paper.title}</strong>
                  {entry.highlights.map((item) => (
                    <p key={item.chunk_id}>
                      p. {item.page_number}: {item.excerpt}
                    </p>
                  ))}
                </article>
              ))
            ) : (
              <p className="empty">
                Select at least two papers and compare them.
              </p>
            )}
          </section>

          <section className="result">
            <h3>
              <Search size={18} /> Related Work
            </h3>
            {related.length ? (
              related.map((item) => (
                <article key={item.chunk_id}>
                  <strong>{item.paper_title}</strong>
                  <span>
                    Page {item.page_number} · score {item.score}
                  </span>
                  <p>{item.excerpt}</p>
                </article>
              ))
            ) : (
              <p className="empty">
                {relatedSearched
                  ? "No related papers found yet. Upload a different paper to compare against this one."
                  : "Choose Related on a paper after uploading more than one PDF."}
              </p>
            )}
          </section>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
