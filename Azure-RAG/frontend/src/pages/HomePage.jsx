import { Link } from 'react-router-dom'
import './ChatPage.scss'

export default function HomePage() {
  return (
    <main>
      <header className="hero">
        <span className="eyebrow">Document intelligence workspace</span>
        <h1>Turn PDFs into useful knowledge</h1>
        <p>Create linked wiki pages from source documents or ask questions across multiple PDFs with RAG.</p>
      </header>

      <section className="home-options" aria-label="Available tools">
        <article className="option-card">
          <span className="option-number">01</span>
          <h2>Create Wiki LLM</h2>
          <p>Generate entity-centric wiki pages and build a reusable knowledge index from multiple PDFs.</p>
          <Link className="primary-link" to="/wiki">Open Wiki Builder</Link>
        </article>

        <article className="option-card">
          <span className="option-number">02</span>
          <h2>RAG Chat</h2>
          <p>Upload one or more PDFs and ask grounded questions directly against their contents.</p>
          <Link className="primary-link" to="/chat">Open RAG Chat</Link>
        </article>
      </section>
    </main>
  )
}
