import { useState } from 'react'
import { sendChatRequest } from '../services/chatService'
import './ChatPage.scss'

export default function ChatPage() {
  const [question, setQuestion] = useState('')
  const [files, setFiles] = useState([])
  const [answer, setAnswer] = useState('')
  const [history, setHistory] = useState([])
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

    try {
      const payload = await sendChatRequest(question, files)
      const responseText = payload.answer || 'No response'
      setAnswer(responseText)
      setHistory((prev) => [
        ...prev,
        { question, answer: responseText },
      ])
      setQuestion('')
    } catch (err) {
      setError(err.message || 'Something went wrong')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <header>
        <h1>Azure RAG Chatbot</h1>
        <p>Upload PDFs and ask a question. The model will use retrieval over the uploaded documents.</p>
      </header>

      <main>
        <form className="chat-form" onSubmit={submitQuestion}>
          <label className="form-label">
            Upload PDF documents
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

          <label className="form-label">
            Ask a question
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Enter your question here"
              rows="5"
            />
          </label>

          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Thinking...' : 'Submit'}
          </button>

          {error && <div className="error">{error}</div>}
        </form>

        <section className="response-panel">
          <h2>Latest answer</h2>
          <pre>{answer || 'No answer yet.'}</pre>
        </section>

        <section className="history-panel">
          <h2>Conversation history</h2>
          {history.length === 0 ? (
            <p>No history yet.</p>
          ) : (
            history.map((item, index) => (
              <div key={index} className="history-item">
                <div className="history-question">Q: {item.question}</div>
                <div className="history-answer">A: {item.answer}</div>
              </div>
            ))
          )}
        </section>
      </main>
    </div>
  )
}
