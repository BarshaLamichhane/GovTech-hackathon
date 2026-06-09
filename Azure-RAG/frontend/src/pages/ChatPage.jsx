import { useState } from 'react'
import { sendChatRequest } from '../services/chatService'
import './ChatPage.scss'

export default function ChatPage() {
  const [question, setQuestion] = useState('')
  const [files, setFiles] = useState([])
  const [messages, setMessages] = useState([])
  const [status, setStatus] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const handleFileChange = (event) => {
    setFiles(Array.from(event.target.files))
  }

  const submitQuestion = async (event) => {
    event.preventDefault()
    if (!question.trim()) {
      return
    }

    setIsLoading(true)
    setError('')
    setStatus(files.length
      ? `Indexing ${files.length} PDF${files.length === 1 ? '' : 's'} and generating an answer...`
      : 'Searching the existing index and generating an answer...')

    try {
      const userMessage = { role: 'user', text: question }
      setMessages((prev) => [...prev, userMessage])

      const payload = await sendChatRequest(question, files)
      const responseText = payload.answer || 'No response'
      const assistantMessage = { role: 'assistant', text: responseText }
      setMessages((prev) => [...prev, assistantMessage])
      setQuestion('')
      setStatus('Answer received.')
    } catch (err) {
      setError(err.message || 'Something went wrong')
      setStatus('')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main>
      <header className="page-header">
        <span className="eyebrow">Retrieval-augmented generation</span>
        <h1>Chat with your PDFs</h1>
        <p>Select multiple PDFs and ask a question grounded in their contents.</p>
      </header>

      <div className="chat-layout">
        <section className="control-panel">
          <div className="upload-panel">
            <h2>Reference documents</h2>
            <p className="small-note">Choose one or more PDFs for this RAG conversation.</p>
            <label className="form-label">
              <span className="label-text">Choose multiple PDF files</span>
              <input type="file" accept="application/pdf" multiple onChange={handleFileChange} />
            </label>
            {files.length > 0 && (
              <div className="file-list">
                <strong>Files ready for upload:</strong>
                <ul>
                  {files.map((file, index) => (
                    <li key={index}>{file.name}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <form className="chat-form" onSubmit={submitQuestion}>
            <label className="form-label">
              Ask a question
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask anything about the uploaded PDFs"
                rows="4"
              />
            </label>
            <button type="submit" disabled={isLoading}>
              {isLoading ? 'Thinking...' : 'Send'}
            </button>
            {error && <div className="error">{error}</div>}
            {status && <div className="status">{status}</div>}
          </form>
        </section>

        <section className="chat-panel">
          <div className="messages">
            {messages.length === 0 ? (
              <div className="empty-state">Your chat will appear here after asking a question.</div>
            ) : (
              messages.map((message, index) => (
                <div key={index} className={`message ${message.role}`}>
                  <div className="message-role">{message.role === 'assistant' ? 'GPT' : message.role === 'user' ? 'You' : 'System'}</div>
                  <div className="message-text">{message.text}</div>
                </div>
              ))
            )}
          </div>
        </section>
      </div>
    </main>
  )
}
