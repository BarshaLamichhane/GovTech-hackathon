const API_URL = 'http://127.0.0.1:8001/api'

const byId = (id) => document.getElementById(id)

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (character) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;',
  })[character])
}

async function request(endpoint, body) {
  const response = await fetch(`${API_URL}/${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const payload = await response.json()
  if (!response.ok) throw new Error(payload.detail || 'Request failed')
  return payload
}

async function uploadPdfs(files) {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))
  const response = await fetch(`${API_URL}/extract-pdfs`, { method: 'POST', body: formData })
  const payload = await response.json()
  if (!response.ok) throw new Error(payload.detail || 'PDF extraction failed')
  return payload
}

function vectorPreview(vector) {
  return vector.length > 18
    ? `${vector.slice(0, 18).join(', ')} ...`
    : vector.join(', ')
}

function renderCountCalculation(chunk, vocabulary) {
  const nonZeroCounts = chunk.word_counts
    .map((count, index) => ({
      word: vocabulary[index],
      count,
      normalized: chunk.vector[index],
    }))
    .filter((item) => item.count > 0)

  const squaredCounts = nonZeroCounts.map((item) => `${item.count}²`).join(' + ')
  const squaredTotal = nonZeroCounts.reduce((total, item) => total + item.count ** 2, 0)

  return `
    <div class="calculation-block">
      <h4>1. Count each vocabulary word in this chunk</h4>
      <p class="calculation-note">Zero-count words are hidden to keep the table readable.</p>
      <div class="table-wrap compact-table">
        <table>
          <thead>
            <tr><th>Word</th><th>Raw count</th><th>Normalized value</th><th>Calculation</th></tr>
          </thead>
          <tbody>
            ${nonZeroCounts.map((item) => `
              <tr>
                <td><strong>${escapeHtml(item.word)}</strong></td>
                <td>${item.count}</td>
                <td>${item.normalized}</td>
                <td><code>${item.count} / ${chunk.raw_vector_magnitude} = ${item.normalized}</code></td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>

      <h4>2. Calculate raw magnitude</h4>
      <div class="formula-walkthrough">
        <code>magnitude = √(count₁² + count₂² + ...)</code>
        <code>magnitude = √(${squaredCounts})</code>
        <code>magnitude = √${squaredTotal} = ${chunk.raw_vector_magnitude}</code>
      </div>

      <h4>3. Normalize every count</h4>
      <p class="calculation-note">
        Divide every raw count by ${chunk.raw_vector_magnitude}. The resulting vector has length 1,
        so long chunks do not win only because they contain more words.
      </p>

      <details class="advanced-vector">
        <summary>Advanced: see complete arrays</summary>
        <code>Counts: [${vectorPreview(chunk.word_counts)}]</code>
        <code>Normalized: [${vectorPreview(chunk.vector)}]</code>
      </details>
    </div>
  `
}

function renderChunks(chunks, vocabulary) {
  byId('chunks').innerHTML = chunks.map((chunk) => `
    <article class="chunk-card">
      <span class="chunk-label">Chunk ${chunk.chunk_id}</span>
      <div class="chunk-meta">Words ${chunk.start_word}-${chunk.end_word}</div>
      <p>${escapeHtml(chunk.text)}</p>
      ${chunk.overlap_from_previous.length
        ? `<div class="overlap"><strong>Repeated overlap:</strong> ${escapeHtml(chunk.overlap_from_previous.join(' '))}</div>`
        : '<div class="overlap empty-overlap">First chunk has no previous overlap</div>'}
      <details>
        <summary>See how counts, magnitude, and normalization are calculated</summary>
        ${renderCountCalculation(chunk, vocabulary)}
      </details>
    </article>
  `).join('')
}

function renderIndexDetails(payload) {
  byId('chunkSummary').innerHTML = `
    <div><strong>${payload.document_word_count}</strong><span>document words</span></div>
    <div><strong>${payload.chunk_size}</strong><span>words per chunk</span></div>
    <div><strong>${payload.chunk_overlap}</strong><span>overlap words</span></div>
    <div><strong>${payload.step_size}</strong><span>words moved each time</span></div>
  `
  byId('indexingSteps').innerHTML = payload.indexing_steps.map((item) => `
    <article class="pipeline-step">
      <span>${item.step}</span>
      <div><strong>${escapeHtml(item.name)}</strong><p>${escapeHtml(item.result)}</p></div>
    </article>
  `).join('')
  byId('vectorStore').innerHTML = payload.vector_store_records.map((record) => `
    <tr>
      <td><code>${record.record_id}</code></td>
      <td><code>${escapeHtml(JSON.stringify(record.metadata))}</code></td>
      <td>${escapeHtml(record.text)}</td>
      <td><code>[${vectorPreview(record.vector)}]</code></td>
    </tr>
  `).join('')
}

function renderSearch(payload) {
  byId('scores').innerHTML = payload.all_similarity_scores.map((item) => `
    <div class="score-card ${item.score >= payload.minimum_similarity ? 'accepted-score' : 'rejected-score'}">
      <div class="score-row">
        <span>Chunk ${item.chunk_id}</span>
        <div class="score-track"><div style="width: ${Math.max(0, item.score * 100)}%"></div></div>
        <strong>${item.score}</strong>
      </div>
      <code>dot product ${item.dot_product} / (${item.question_vector_magnitude} × ${item.chunk_vector_magnitude}) = ${item.score}</code>
      <small>${item.score >= payload.minimum_similarity ? 'Accepted by threshold' : 'Rejected by threshold'}</small>
    </div>
  `).join('')

  byId('retrieved').innerHTML = payload.retrieved_chunks.map((chunk) => `
    <article class="chunk-card retrieved-card">
      <span class="chunk-label">Chunk ${chunk.chunk_id} · score ${chunk.score}</span>
      <p>${escapeHtml(chunk.text)}</p>
    </article>
  `).join('') || '<p class="no-context">No chunks passed the relevance threshold, so no context will be sent to an LLM.</p>'
  byId('retrievalDecision').className = `decision ${payload.has_relevant_context ? 'decision-accept' : 'decision-reject'}`
  byId('retrievalDecision').innerHTML = `
    <strong>${payload.has_relevant_context ? 'Relevant context accepted' : 'No relevant context found'}</strong>
    <p>${escapeHtml(payload.retrieval_decision)}</p>
    <small>Nearest-neighbor search still returned candidates. The threshold decides whether they are relevant enough to use.</small>
  `
  byId('questionSteps').innerHTML = payload.question_embedding_steps.map((text, index) => `
    <article class="pipeline-step">
      <span>${index + 1}</span>
      <div><strong>${escapeHtml(text)}</strong></div>
    </article>
  `).join('')
  const nonZeroDimensions = payload.question_word_counts.filter((value) => value > 0).length
  byId('questionSummary').innerHTML = `
    <div><strong>${payload.question_tokens.length}</strong><span>question tokens</span></div>
    <div><strong>${payload.vocabulary.length}</strong><span>vector dimensions</span></div>
    <div><strong>${nonZeroDimensions}</strong><span>non-zero dimensions</span></div>
    <div><strong>${payload.question_raw_magnitude}</strong><span>raw vector magnitude</span></div>
  `
  byId('questionEmbedding').innerHTML = payload.vocabulary.map((word, index) => `
    <tr class="${payload.question_word_counts[index] > 0 ? 'active-dimension' : ''}">
      <td>${index}</td>
      <td>${escapeHtml(word)}</td>
      <td>${payload.question_word_counts[index]}</td>
      <td>${payload.question_vector[index]}</td>
    </tr>
  `).join('')
  byId('context').textContent = payload.context_sent_to_llm
  byId('answer').textContent = payload.simple_extractive_answer
  byId('searchResults').classList.remove('hidden')
}

byId('indexButton').addEventListener('click', async () => {
  byId('indexError').textContent = ''
  try {
    const payload = await request('index', {
      text: byId('documentText').value,
      chunk_size: Number(byId('chunkSize').value),
      chunk_overlap: Number(byId('chunkOverlap').value),
    })
    byId('vocabulary').innerHTML = payload.vocabulary
      .map((word, index) => `<span title="Vector position ${index}">${index}: ${word}</span>`)
      .join('')
    renderChunks(payload.chunks, payload.vocabulary)
    renderIndexDetails(payload)
    byId('indexResults').classList.remove('hidden')
    byId('searchResults').classList.add('hidden')
  } catch (error) {
    byId('indexError').textContent = error.message
  }
})

const defaultDocumentText = byId('documentText').value

byId('defaultTextButton').addEventListener('click', () => {
  byId('documentText').value = defaultDocumentText
  byId('sourceStatus').textContent = 'Current source: default example text'
  byId('pdfSummary').classList.add('hidden')
  byId('indexError').textContent = ''
})

byId('documentText').addEventListener('input', () => {
  byId('sourceStatus').textContent = 'Current source: typed, pasted, or edited text'
})

byId('uploadPdfButton').addEventListener('click', async () => {
  byId('indexError').textContent = ''
  const files = Array.from(byId('pdfFiles').files)
  if (!files.length) {
    byId('indexError').textContent = 'Choose at least one PDF before extracting text.'
    return
  }
  try {
    byId('sourceStatus').textContent = `Extracting text from ${files.length} PDF${files.length === 1 ? '' : 's'}...`
    const payload = await uploadPdfs(files)
    byId('documentText').value = payload.combined_text
    byId('sourceStatus').textContent = `Current source: ${payload.file_count} PDF${payload.file_count === 1 ? '' : 's'} · ${payload.total_pages_with_text}/${payload.total_pages} pages contained selectable text`
    byId('pdfSummary').innerHTML = payload.files.map((file) => `
      <div><strong>${escapeHtml(file.filename)}</strong><span>${file.pages_with_text}/${file.page_count} pages with text</span></div>
    `).join('')
    byId('pdfSummary').classList.remove('hidden')
  } catch (error) {
    byId('sourceStatus').textContent = 'PDF extraction failed'
    byId('indexError').textContent = error.message
  }
})

byId('searchButton').addEventListener('click', async () => {
  byId('searchError').textContent = ''
  try {
    const payload = await request('search', {
      question: byId('question').value,
      top_k: Number(byId('topK').value),
      minimum_similarity: Number(byId('minimumSimilarity').value),
    })
    renderSearch(payload)
  } catch (error) {
    byId('searchError').textContent = error.message
  }
})
