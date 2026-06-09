const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api'

async function postFormData(endpoint, formData) {
  const response = await fetch(`${API_URL}/${endpoint}`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const contentType = response.headers.get('content-type') || ''
    let errorText = 'Backend request failed'
    if (contentType.includes('application/json')) {
      const body = await response.json()
      errorText = body.detail || JSON.stringify(body)
    } else {
      errorText = await response.text()
    }
    throw new Error(errorText || 'Backend request failed')
  }

  return response.json()
}

export async function sendChatRequest(question, files) {
  const formData = new FormData()
  formData.append('question', question)

  files.forEach((file) => {
    formData.append('files', file)
  })

  return postFormData('chat', formData)
}

export async function buildKnowledgeBase(files) {
  const formData = new FormData()

  files.forEach((file) => {
    formData.append('files', file)
  })

  return postFormData('build-knowledge-base', formData)
}

export async function getKnowledgeBaseStatus() {
  const response = await fetch(`${API_URL}/knowledge-base/status`)
  if (!response.ok) {
    const body = await response.text()
    throw new Error(body || 'Could not fetch knowledge base status')
  }
  return response.json()
}
