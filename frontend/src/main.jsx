import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import {
  BookOpen,
  Cpu,
  FileText,
  GitCompare,
  History,
  Loader2,
  MessageSquareText,
  Plus,
  RefreshCw,
  Search,
  Sparkles,
  Trash2,
  Upload,
} from "lucide-react";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function api(path, options) {
  let response;
  try {
    response = await fetch(`${API}${path}`, options);
  } catch {
    throw new Error(`Cannot reach the backend at ${API}. Is uvicorn running?`);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join(", ") : body.detail;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function Markdown({ children }) {
  return <div className="markdown"><ReactMarkdown>{children || ""}</ReactMarkdown></div>;
}

function Citations({ citations }) {
  if (!citations?.length) return null;
  return (
    <details className="citations">
      <summary>{citations.length} cited excerpts</summary>
      {citations.map((citation, index) => (
        <article key={`${citation.chunk_id}-${index}`}>
          <strong>
            [{index + 1}] {citation.paper_title}
          </strong>
          <span>
            Page {citation.page_number} · relevance {citation.score}
          </span>
          <p>{citation.excerpt}</p>
        </article>
      ))}
    </details>
  );
}

function App() {
  const [health, setHealth] = useState(null);
  const [papers, setPapers] = useState([]);
  const [chats, setChats] = useState([]);
  const [question, setQuestion] = useState("");
  const [chatId, setChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [selected, setSelected] = useState([]);
  const [summary, setSummary] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [related, setRelated] = useState(null);
  const threadEnd = useRef(null);

  const hasPapers = papers.length > 0;

  async function refresh() {
    const [paperData, chatData] = await Promise.all([api("/papers"), api("/chats")]);
    setPapers(paperData);
    setChats(chatData);
  }

  useEffect(() => {
    api("/health").then(setHealth).catch((err) => setError(err.message));
    refresh().catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    threadEnd.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [messages]);

  async function run(label, task) {
    setBusy(label);
    setError("");
    try {
      await task();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function upload(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    await run("upload", async () => {
      const problems = [];
      for (const file of files) {
        const form = new FormData();
        form.append("file", file);
        try {
          await api("/papers", { method: "POST", body: form });
        } catch (err) {
          problems.push(`${file.name}: ${err.message}`);
        }
      }
      await refresh();
      if (problems.length) setError(problems.join(" · "));
    });
  }

  async function ask() {
    const text = question.trim();
    if (text.length < 3) return;
    await run("ask", async () => {
      setMessages((current) => [...current, { role: "user", content: text, pending: true }]);
      setQuestion("");
      try {
        const data = await api("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: text, chat_id: chatId, paper_ids: selected.length ? selected : null }),
        });
        setChatId(data.chat_id);
        setMessages((current) => [
          ...current.map((m) => ({ ...m, pending: false })),
          { role: "assistant", content: data.answer, citations: data.citations, llm: data.llm },
        ]);
        await refresh();
      } catch (err) {
        setMessages((current) => current.filter((m) => !m.pending));
        setQuestion(text);
        throw err;
      }
    });
  }

  async function openChat(id) {
    await run(`chat-${id}`, async () => {
      const data = await api(`/chats/${id}`);
      setChatId(id);
      setMessages(data.messages);
    });
  }

  function newChat() {
    setChatId(null);
    setMessages([]);
  }

  async function deleteChat(id) {
    await run(`delete-chat-${id}`, async () => {
      await api(`/chats/${id}`, { method: "DELETE" });
      if (id === chatId) newChat();
      await refresh();
    });
  }

  async function loadSummary(paper, refreshSummary = false) {
    await run(`summary-${paper.id}`, async () => {
      setSummary({ title: paper.title, paperId: paper.id, text: "", loading: true });
      const data = await api(`/papers/${paper.id}/summary${refreshSummary ? "?refresh=true" : ""}`);
      setSummary({ title: paper.title, paperId: paper.id, text: data.summary, llm: data.llm });
    });
  }

  async function compare() {
    await run("compare", async () => {
      const data = await api("/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paper_ids: selected }),
      });
      setComparison(data.comparison);
    });
  }

  async function findRelated(paper, refresh = false) {
    await run(`related-${paper.id}`, async () => {
      const data = await api("/related", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paper_id: paper.id, refresh }),
      });
      setRelated({
        paper,
        title: paper.title,
        library: data.library || [],
        external: data.external || [],
        source: data.source,
        notice: data.notice,
        keywords: data.keywords || [],
      });
    });
  }

  async function deletePaper(paper) {
    if (!window.confirm(`Delete "${paper.title}"?`)) return;
    await run(`delete-${paper.id}`, async () => {
      await api(`/papers/${paper.id}`, { method: "DELETE" });
      setSelected((current) => current.filter((id) => id !== paper.id));
      setComparison(null);
      setRelated(null);
      if (summary?.paperId === paper.id) setSummary(null);
      await refresh();
    });
  }

  const selectedTitles = useMemo(
    () => papers.filter((paper) => selected.includes(paper.id)).map((paper) => paper.title),
    [papers, selected],
  );

  const llm = health?.llm;

  return (
    <main>
      <aside>
        <div className="brand">
          <BookOpen size={26} />
          <div>
            <h1>ResearchAI</h1>
            <p>Literature review workspace</p>
          </div>
        </div>

        {health && (
          <div className={`status ${llm?.enabled ? "ok" : "warn"}`}>
            <Cpu size={15} />
            {llm?.enabled ? (
              <span>
                {llm.provider} · {llm.model}
              </span>
            ) : (
              <span>No LLM key: extractive answers only</span>
            )}
            {!health.embeddings && <span> · keyword search</span>}
          </div>
        )}

        <label className={`upload ${busy === "upload" ? "busy" : ""}`}>
          {busy === "upload" ? <Loader2 className="spin" size={22} /> : <Upload size={22} />}
          <span>{busy === "upload" ? "Indexing PDFs..." : "Upload PDFs"}</span>
          <input
            type="file"
            accept="application/pdf"
            multiple
            disabled={busy === "upload"}
            onChange={(e) => {
              upload(e.target.files);
              e.target.value = "";
            }}
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
                      event.target.checked ? [...current, paper.id] : current.filter((id) => id !== paper.id),
                    )
                  }
                />
                <span>{paper.title}</span>
              </label>
              <small>
                {paper.page_count} pages · {paper.chunk_count} chunks
              </small>
              <div className="row">
                <button disabled={busy.startsWith("summary")} onClick={() => loadSummary(paper)}>
                  Summary
                </button>
                <button disabled={busy.startsWith("related")} onClick={() => findRelated(paper)}>
                  Related
                </button>
                <button
                  className="danger"
                  disabled={busy === `delete-${paper.id}`}
                  title="Delete paper"
                  onClick={() => deletePaper(paper)}
                >
                  <Trash2 size={15} />
                </button>
              </div>
            </div>
          ))}
          {!hasPapers && <p className="empty">Upload PDFs to build your local research index.</p>}
        </section>

        <section className="panel">
          <h2>
            <History size={17} /> Chats
            <button className="icon-button" title="New chat" onClick={newChat}>
              <Plus size={16} />
            </button>
          </h2>
          {chats.slice(0, 12).map((chat) => (
            <div className={`chat-item ${chat.id === chatId ? "active" : ""}`} key={chat.id}>
              <button className="chat-link" onClick={() => openChat(chat.id)} title={chat.title}>
                {chat.title}
              </button>
              <button className="icon-button" title="Delete chat" onClick={() => deleteChat(chat.id)}>
                <Trash2 size={14} />
              </button>
            </div>
          ))}
          {!chats.length && <p className="empty">Saved conversations will appear here.</p>}
        </section>
      </aside>

      <section className="workspace">
        <div className="topbar">
          <div>
            <h2>Ask, compare, and cite your papers</h2>
            <p>
              {selectedTitles.length
                ? `Questions and comparison use: ${selectedTitles.join(", ")}`
                : "Questions search all papers. Tick papers to focus on them or to compare."}
            </p>
          </div>
          <button className="primary" disabled={selected.length < 2 || busy === "compare"} onClick={compare}>
            {busy === "compare" ? <Loader2 className="spin" size={18} /> : <GitCompare size={18} />}
            Compare
          </button>
        </div>

        {error && (
          <div className="error" onClick={() => setError("")}>
            {error}
          </div>
        )}

        <section className="result thread">
          <h3>
            <Sparkles size={18} /> Conversation
            {chatId && <small className="muted">chat #{chatId}</small>}
          </h3>
          {messages.length ? (
            messages.map((message, index) => (
              <div className={`message ${message.role}`} key={message.id || index}>
                {message.role === "user" ? (
                  <p>{message.content}</p>
                ) : (
                  <>
                    <Markdown>{message.content}</Markdown>
                    <Citations citations={message.citations} />
                  </>
                )}
              </div>
            ))
          ) : (
            <p className="empty">Ask a question to get an answer grounded in your papers, with page citations.</p>
          )}
          {busy === "ask" && (
            <div className="message assistant">
              <Loader2 className="spin" size={18} /> Searching papers and writing an answer...
            </div>
          )}
          <div ref={threadEnd} />
        </section>

        <section className="ask">
          <MessageSquareText size={22} />
          <textarea
            placeholder="Ask a question, e.g. What limitations do these papers report?  (Ctrl/Cmd + Enter to send)"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) ask();
            }}
          />
          <button className="primary" disabled={!hasPapers || busy === "ask" || question.trim().length < 3} onClick={ask}>
            {busy === "ask" ? <Loader2 className="spin" size={18} /> : <Search size={18} />}
            Ask
          </button>
        </section>

        <div className="grid">
          <section className="result">
            <h3>
              <FileText size={18} /> Summary
              {summary?.text && (
                <button
                  className="icon-button dark"
                  title="Regenerate summary"
                  onClick={() => loadSummary({ id: summary.paperId, title: summary.title }, true)}
                >
                  <RefreshCw size={14} />
                </button>
              )}
            </h3>
            {summary ? (
              summary.loading ? (
                <p className="empty">
                  <Loader2 className="spin" size={16} /> Summarising {summary.title}...
                </p>
              ) : (
                <>
                  <p className="muted">{summary.title}</p>
                  <Markdown>{summary.text}</Markdown>
                </>
              )
            ) : (
              <p className="empty">Choose Summary on any paper.</p>
            )}
          </section>

          <section className="result">
            <h3>
              <Search size={18} /> Related Work
            </h3>
            {related ? (
              <>
                <div className="related-head">
                  <p className="muted">
                    Published papers related to <strong>{related.title}</strong>
                    {related.source ? ` · via ${related.source}` : ""}
                  </p>
                  <button
                    className="ghost"
                    disabled={busy.startsWith("related")}
                    onClick={() => findRelated(related.paper, true)}
                    title="Search the paper databases again"
                  >
                    <RefreshCw size={14} /> Refresh
                  </button>
                </div>
                {related.notice ? <p className="notice">{related.notice}</p> : null}
                {related.external.map((item, index) => (
                  <article key={`${item.url || item.title}-${index}`} className="related-paper">
                    <strong>
                      {item.url ? (
                        <a href={item.url} target="_blank" rel="noreferrer">
                          {item.title}
                        </a>
                      ) : (
                        item.title
                      )}
                    </strong>
                    <span>
                      {[item.authors, item.venue, item.year].filter(Boolean).join(" · ")}
                      {item.citations != null ? ` · ${item.citations.toLocaleString()} citations` : ""}
                      {item.in_library ? " · in your library" : ""}
                    </span>
                    {item.reason ? (
                      <p className="reason">
                        {item.relation ? <em className="tag">{item.relation}</em> : null} {item.reason}
                      </p>
                    ) : null}
                    {item.abstract ? <p className="abstract">{item.abstract}</p> : null}
                    {item.pdf_url ? (
                      <a className="pdf-link" href={item.pdf_url} target="_blank" rel="noreferrer">
                        Open PDF
                      </a>
                    ) : null}
                  </article>
                ))}
                <h4 className="subhead">In your library</h4>
                {related.library.length ? (
                  related.library.map((item) => (
                    <article key={item.chunk_id} className="library-match">
                      <strong>{item.paper_title}</strong>
                      <span>Similarity score {item.score.toFixed(2)} (embedding cosine)</span>
                    </article>
                  ))
                ) : (
                  <p className="empty">Upload more PDFs to see which of your papers are closest to this one.</p>
                )}
              </>
            ) : (
              <p className="empty">Choose Related on any paper to find similar published work.</p>
            )}
          </section>
        </div>

        <section className="result">
          <h3>
            <GitCompare size={18} /> Comparison
          </h3>
          {comparison?.aspects?.length ? (
            <>
              {comparison.overview && <Markdown>{comparison.overview}</Markdown>}
              {comparison.aspects.map((aspect) => (
                <article key={aspect.label} className="comparison-aspect">
                  <strong>{aspect.label}</strong>
                  <div className="comparison-points">
                    {aspect.points.map((point) => (
                      <div className="comparison-point" key={`${aspect.label}-${point.paper_id}`}>
                        <span>{point.paper_title}</span>
                        <p>{point.text}</p>
                        {point.page_number && <small>Source: page {point.page_number}</small>}
                      </div>
                    ))}
                  </div>
                </article>
              ))}
            </>
          ) : (
            <p className="empty">Tick at least two papers and press Compare.</p>
          )}
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
