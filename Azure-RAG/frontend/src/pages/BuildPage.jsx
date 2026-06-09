import { useState } from 'react'
import { buildKnowledgeBase } from '../services/chatService'
import './ChatPage.scss'

export default function BuildPage() {
  const [files, setFiles] = useState([])
  const [status, setStatus] = useState('')
  const [generatedPages, setGeneratedPages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const handleFileChange = (event) => setFiles(Array.from(event.target.files))

  const handleBuild = async () => {
    if (!files.length) {
      setError('Select at least one PDF to build the knowledge base.')
      return
    }
    setIsLoading(true)
    setError('')
    setStatus('Uploading and building knowledge base...')
    try {
      const payload = await buildKnowledgeBase(files)
      setStatus(payload.message || 'Done')
      setGeneratedPages(payload.generated_pages || [])
    } catch (err) {
      setError(err.message || 'Build failed')
      setStatus('')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main>
      <header className="page-header">
        <span className="eyebrow">Wiki LLM</span>
        <h1>Create a linked knowledge base</h1>
        <p>Upload multiple PDFs to generate entity-centric wiki pages and index them for later use.</p>
      </header>

      <section className="single-panel">
        <section className="control-panel">
          <div className="upload-panel">
            <label className="form-label">
              <span className="label-text">Choose PDF files</span>
              <input type="file" accept="application/pdf" multiple onChange={handleFileChange} />
            </label>
            {files.length > 0 && (
              <div className="file-list">
                <strong>Files ready for upload:</strong>
                <ul>
                  {files.map((file, idx) => (
                    <li key={idx}>{file.name}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="action-buttons">
              <button onClick={handleBuild} disabled={isLoading}>
                {isLoading ? 'Creating Wiki LLM...' : 'Create Wiki LLM'}
              </button>
            </div>
            {status && <div className="status">{status}</div>}
            {error && <div className="error">{error}</div>}
            {generatedPages.length > 0 && (
              <div className="existing-pages">
                <strong>Generated wiki pages:</strong>
                <ul>
                  {generatedPages.map((page) => <li key={page}>{page}</li>)}
                </ul>
              </div>
            )}
          </div>
        </section>
      </section>
    </main>
  )
}
