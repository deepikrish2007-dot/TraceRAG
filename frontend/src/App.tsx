import { useEffect, useMemo, useRef, useState, type ChangeEvent, type FormEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import type { Plugin } from "unified";
import type { Root, Text } from "mdast";
import {
  LayoutDashboard,
  Files,
  MessageSquareText,
  Settings,
  Search,
  Upload,
  Menu,
  FileText,
  Trash2,
  Send,
  LoaderCircle,
  CheckCircle2,
  AlertCircle,
  Bot,
  User,
  RefreshCw,
  X,
  BookOpen,
  Sparkles,
} from "lucide-react";
import "./App.css";
import "katex/dist/katex.min.css";

const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://127.0.0.1:8000";

type DocumentItem = {
  filename: string;
  size_bytes?: number;
  size_kb?: number;
  [key: string]: unknown;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

/** Turn supported source/page citations into clickable PDF links. */
const remarkCitationBadges: Plugin<[], Root> = () => (tree) => {
  const citationPattern = /\(Source:\s*([^,\n]+?\.pdf),\s*p\.?\s*(\d+)\)|\[([^\]\n]+?\.pdf),\s*Page\s*(\d+)\]/gi;

  const visit = (node: any) => {
    if (!Array.isArray(node.children)) return;
    const updated: any[] = [];

    for (const child of node.children) {
      if (child.type === "text" && typeof child.value === "string") {
        const value = child.value as string;
        let lastIndex = 0;
        let match: RegExpExecArray | null;
        citationPattern.lastIndex = 0;

        while ((match = citationPattern.exec(value)) !== null) {
          if (match.index > lastIndex) {
            updated.push({ type: "text", value: value.slice(lastIndex, match.index) });
          }
          const filename = (match[1] ?? match[3] ?? "").trim();
          const page = Number(match[2] ?? match[4]);
          const href = `${API_URL}/documents/${encodeURIComponent(filename)}/view?page=${page}#page=${page}`;
          updated.push({
            type: "link",
            url: href,
            title: `Open ${filename} at page ${page}`,
            children: [{ type: "text", value: `${filename} · p. ${page}` }],
          });
          lastIndex = match.index + match[0].length;
        }

        if (lastIndex < value.length) {
          updated.push({ type: "text", value: value.slice(lastIndex) });
        }
        if (lastIndex === 0 && value.length === 0) updated.push(child);
      } else {
        visit(child);
        updated.push(child);
      }
    }
    node.children = updated;
  };

  visit(tree);
};

function formatFileSize(bytes?: number): string {
  if (!bytes || !Number.isFinite(bytes)) return "PDF document";
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function App() {
  const [activePage, setActivePage] = useState("Overview");
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [loadingDocuments, setLoadingDocuments] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [deletingDocument, setDeletingDocument] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [indexedChunks, setIndexedChunks] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const navigation = [
    { name: "Overview", icon: LayoutDashboard },
    { name: "Documents", icon: Files },
    { name: "Research Chat", icon: MessageSquareText },
    { name: "Settings", icon: Settings },
  ];

  async function loadDocuments(showError = true) {
    setLoadingDocuments(true);
    if (showError) setError("");
    try {
      const response = await fetch(`${API_URL}/documents`);
      if (!response.ok) throw new Error(`Could not load documents (${response.status}).`);
      const data = await response.json();
      const items = Array.isArray(data.documents) ? data.documents : [];
      setDocuments(
        items.map((doc: string | Record<string, unknown>) =>
          typeof doc === "string"
            ? { filename: doc }
            : { ...doc, filename: String(doc.filename ?? doc.name ?? "Unknown document") }
        )
      );
      if (typeof data.indexed_chunks === "number") setIndexedChunks(data.indexed_chunks);
    } catch (err) {
      if (showError) {
        setError(err instanceof Error ? err.message : "Cannot connect to the TraceRAG backend. Check that FastAPI is running on port 8000.");
      }
    } finally {
      setLoadingDocuments(false);
    }
  }

  useEffect(() => {
    void loadDocuments();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, asking]);

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Please select a PDF file.");
      event.target.value = "";
      return;
    }

    setUploading(true);
    setNotice("");
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`${API_URL}/upload`, { method: "POST", body: formData });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || data.message || "PDF upload failed.");
      setNotice(data.message || `${file.name} uploaded and processed successfully.`);
      await loadDocuments(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong during upload.");
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  async function handleDelete(filename: string) {
    if (!window.confirm(`Are you sure you want to delete "${filename}"?`)) return;
    setDeletingDocument(filename);
    setNotice("");
    setError("");
    try {
      const response = await fetch(`${API_URL}/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Could not delete this document.");
      setNotice(data.message || `${filename} deleted successfully.`);
      await loadDocuments(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Document deletion failed.");
    } finally {
      setDeletingDocument(null);
    }
  }

  async function handleAsk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const userQuestion = question.trim();
    if (!userQuestion || asking) return;

    setMessages((previous) => [...previous, { role: "user", content: userQuestion }]);
    setQuestion("");
    setAsking(true);
    setError("");
    setActivePage("Research Chat");

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userQuestion }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || `Could not get an answer (${response.status}).`);
      const answer = data.answer ?? data.response ?? data.result;
      if (answer === undefined || answer === null) throw new Error("The backend response did not contain an answer field.");
      setMessages((previous) => [
        ...previous,
        { role: "assistant", content: typeof answer === "string" ? answer : JSON.stringify(answer, null, 2) },
      ]);
    } catch (err) {
      setMessages((previous) => [
        ...previous,
        { role: "assistant", content: err instanceof Error ? `I couldn't retrieve an answer: ${err.message}` : "I couldn't retrieve an answer. Please try again." },
      ]);
    } finally {
      setAsking(false);
    }
  }

  const filteredDocuments = useMemo(
    () => documents.filter((doc) => doc.filename.toLowerCase().includes(searchTerm.toLowerCase())),
    [documents, searchTerm]
  );

  function openChatWithQuestion(value: string) {
    setQuestion(value);
    setActivePage("Research Chat");
  }

  return (
    <div className="app-layout">
      <aside className={`sidebar ${mobileNavOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-icon">T</div>
          <span>TraceRAG</span>
          <button className="mobile-close" onClick={() => setMobileNavOpen(false)} aria-label="Close navigation"><X size={18} /></button>
        </div>
        <p className="nav-label">WORKSPACE</p>
        <nav>
          {navigation.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.name}
                className={`nav-item ${activePage === item.name ? "active" : ""}`}
                onClick={() => { setActivePage(item.name); setMobileNavOpen(false); }}
              >
                <Icon size={18} />
                <span>{item.name}</span>
              </button>
            );
          })}
        </nav>
        <div className="sidebar-footer">
          <span className="status-dot" /> Local workspace
        </div>
      </aside>

      {mobileNavOpen && <button className="sidebar-backdrop" aria-label="Close navigation" onClick={() => setMobileNavOpen(false)} />}

      <main className="main-content">
        <header className="topbar">
          <button className="menu-button" onClick={() => setMobileNavOpen(true)} aria-label="Open navigation"><Menu size={21} /></button>
          <div className="breadcrumb">Workspace <span>/</span> {activePage}</div>
          <div className="topbar-right"><span className="workspace-label">LOCAL WORKSPACE</span><span className="status-dot" /></div>
        </header>

        <section className="content-area">
          {(notice || error) && (
            <div className={`app-alert ${error ? "app-alert-error" : "app-alert-success"}`}>
              {error ? <AlertCircle size={17} /> : <CheckCircle2 size={17} />}
              <span>{error || notice}</span>
              <button onClick={() => { setError(""); setNotice(""); }} aria-label="Dismiss message"><X size={16} /></button>
            </div>
          )}

          {activePage === "Overview" && (
            <div className="overview-panel page-panel">
              <div className="hero-section">
                <div className="eyebrow"><Sparkles size={14} /> YOUR AI RESEARCH ASSISTANT</div>
                <h1>Your research, understood.</h1>
                <p>Explore your documents, ask questions, and discover answers grounded in your research.</p>
                <button className="primary-button" onClick={() => setActivePage("Research Chat")}><MessageSquareText size={17} /> Open Research Chat</button>
              </div>

              <div className="stats-grid">
                <div className="stat-card"><div className="stat-icon"><Files size={19} /></div><span>PDF documents</span><strong>{loadingDocuments ? "…" : documents.length}</strong></div>
                <div className="stat-card"><div className="stat-icon"><BookOpen size={19} /></div><span>Indexed chunks</span><strong>{indexedChunks ?? "—"}</strong></div>
                <div className="stat-card"><div className="stat-icon"><MessageSquareText size={19} /></div><span>Questions this session</span><strong>{messages.filter((message) => message.role === "user").length}</strong></div>
              </div>

              <div className="overview-grid">
                <div className="documents-panel">
                  <div className="panel-heading"><div><h2>Your documents</h2><p>Recently available research PDFs</p></div><button className="icon-button" onClick={() => void loadDocuments()} aria-label="Refresh documents"><RefreshCw size={17} /></button></div>
                  {loadingDocuments ? <div className="loading-state"><LoaderCircle className="spin" size={20} /> Loading documents…</div> : documents.length === 0 ? <div className="empty-state"><FileText size={27} /><h3>No documents yet</h3><p>Upload your first PDF to begin.</p></div> : (
                    <div className="document-list compact-list">{documents.slice(0, 4).map((doc) => <div className="document-row" key={doc.filename}><div className="file-icon"><FileText size={19} /></div><div className="file-details"><strong>{doc.filename}</strong><span>{formatFileSize(typeof doc.size_bytes === "number" ? doc.size_bytes : undefined)}</span></div></div>)}</div>
                  )}
                  <button className="text-button" onClick={() => setActivePage("Documents")}>View all documents <span>→</span></button>
                </div>
                <div className="documents-panel quick-question-panel"><div className="panel-heading"><div><h2>Start exploring</h2><p>Try a question about your research</p></div><Sparkles size={20} /></div>
                  <button className="suggested-question" onClick={() => openChatWithQuestion("Summarize the main contributions of my uploaded research papers.")}>Summarize the main contributions of my papers <span>→</span></button>
                  <button className="suggested-question" onClick={() => openChatWithQuestion("Compare the methods used in my uploaded research papers.")}>Compare the methods used in my papers <span>→</span></button>
                  <button className="suggested-question" onClick={() => openChatWithQuestion("What are the key findings and limitations discussed in my documents?")}>Find key findings and limitations <span>→</span></button>
                </div>
              </div>
            </div>
          )}

          {activePage === "Documents" && (
            <div className="documents-panel page-panel">
              <div className="panel-heading"><div><h2>Documents</h2><p>Upload and manage your indexed research papers.</p></div><Files size={21} className="search-icon" /></div>
              <div className="upload-card">
                <div className="upload-icon"><Upload size={23} /></div>
                <h3>{uploading ? "Uploading and indexing…" : "Add a research paper"}</h3>
                <p>Select a PDF document to upload it to your local TraceRAG library.</p>
                <input ref={fileInputRef} type="file" accept="application/pdf,.pdf" onChange={handleUpload} hidden />
                <button className="primary-button" disabled={uploading} onClick={() => fileInputRef.current?.click()}>{uploading ? <LoaderCircle className="spin" size={17} /> : <Upload size={17} />} {uploading ? "Please wait" : "Choose PDF"}</button>
                <span className="upload-hint">PDF files only · Maximum size depends on backend configuration</span>
              </div>
              <div className="document-toolbar"><div><h3>Library <span className="count-pill">{documents.length}</span></h3></div><div className="search-field"><Search size={17} /><input value={searchTerm} onChange={(event) => setSearchTerm(event.target.value)} placeholder="Search documents…" /></div><button className="secondary-button" onClick={() => void loadDocuments()} disabled={loadingDocuments}><RefreshCw size={16} /> Refresh</button></div>
              {loadingDocuments ? <div className="loading-state"><LoaderCircle className="spin" size={20} /> Loading documents…</div> : filteredDocuments.length === 0 ? <div className="empty-state"><FileText size={28} /><h3>{documents.length === 0 ? "No documents yet" : "No matching documents"}</h3><p>{documents.length === 0 ? "Upload your first PDF to begin." : "Try a different search term."}</p></div> : (
                <div className="document-list">{filteredDocuments.map((doc) => <div className="document-row" key={doc.filename}><div className="file-icon"><FileText size={20} /></div><div className="file-details"><strong>{doc.filename}</strong><span>{formatFileSize(typeof doc.size_bytes === "number" ? doc.size_bytes : undefined)}</span></div><a className="icon-button" href={`${API_URL}/documents/${encodeURIComponent(doc.filename)}/view?page=1#page=1`} target="_blank" rel="noreferrer" title={`Open ${doc.filename}`}><BookOpen size={17} /></a><button className="delete-button" aria-label={`Delete ${doc.filename}`} disabled={deletingDocument === doc.filename} onClick={() => void handleDelete(doc.filename)}>{deletingDocument === doc.filename ? <LoaderCircle size={17} className="spin" /> : <Trash2 size={17} />}</button></div>)}</div>
              )}
            </div>
          )}

          {activePage === "Research Chat" && (
            <div className="documents-panel page-panel chat-panel">
              <div className="chat-heading"><div><div className="eyebrow">RESEARCH WORKSPACE</div><h1>Research Chat</h1><p>Ask questions about your indexed PDF documents.</p></div><div className="chat-heading-icon"><MessageSquareText size={22} /></div></div>
              <div className="chat-messages">
                {messages.length === 0 ? <div className="empty-state chat-empty"><div className="empty-icon"><Bot size={27} /></div><h3>What would you like to understand?</h3><p>Ask a question about your documents to get started.</p><div className="chat-suggestions"><button onClick={() => setQuestion("Summarize the main contributions of my research papers.")}>Summarize my papers</button><button onClick={() => setQuestion("Compare the methods used in my research papers.")}>Compare methods</button></div></div> : messages.map((message, index) => (
                  <div className={`message ${message.role === "user" ? "user-message" : ""}`} key={`${message.role}-${index}`}>
                    <div className="message-avatar">{message.role === "user" ? <User size={17} /> : <Bot size={17} />}</div>
                    <div className="message-content"><strong>{message.role === "user" ? "You" : "TraceRAG"}</strong><div className="markdown-content"><ReactMarkdown remarkPlugins={[remarkMath, remarkGfm, remarkCitationBadges]} rehypePlugins={[rehypeKatex]}>{message.content}</ReactMarkdown></div></div>
                  </div>
                ))}
                {asking && <div className="thinking-indicator"><LoaderCircle size={17} className="spin" /> Searching your research…</div>}
                <div ref={chatEndRef} />
              </div>
              <form className="chat-form" onSubmit={handleAsk}><input type="text" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a question about your documents…" disabled={asking} /><button className="primary-button send-button" type="submit" disabled={asking || !question.trim()} aria-label="Send question">{asking ? <LoaderCircle className="spin" size={17} /> : <Send size={17} />} Ask</button></form>
              <p className="chat-note">Answers are grounded in the documents retrieved by your local TraceRAG backend.</p>
            </div>
          )}

          {activePage === "Settings" && (
            <div className="documents-panel page-panel">
              <div className="panel-heading"><div><h2>Settings</h2><p>Check your local TraceRAG connection.</p></div><Settings size={21} className="search-icon" /></div>
              <div className="settings-list">
                <div className="setting-row"><div><strong>Backend API</strong><p>{API_URL}</p></div><span className="connection-badge">Local</span></div>
                <div className="setting-row"><div><strong>Documents indexed</strong><p>PDF files available in your library</p></div><strong>{documents.length}</strong></div>
                <div className="setting-row"><div><strong>Indexed chunks</strong><p>Chunks stored in the ChromaDB collection</p></div><strong>{indexedChunks ?? "—"}</strong></div>
                <div className="setting-row"><div><strong>Refresh document library</strong><p>Fetch the latest document list from FastAPI</p></div><button className="secondary-button" onClick={() => void loadDocuments()} disabled={loadingDocuments}><RefreshCw size={16} /> Refresh</button></div>
              </div>
              <div className="settings-help"><strong>Math and citation rendering</strong><p>Research Chat renders Markdown, GitHub-style tables, LaTeX equations with KaTeX, and clickable PDF page citations.</p></div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
