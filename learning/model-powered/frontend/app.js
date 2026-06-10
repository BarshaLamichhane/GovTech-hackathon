const API = 'http://127.0.0.1:8002/api'
const el = (id) => document.getElementById(id)
const esc = (value) => String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' })[c])
let sourceDocuments = []
let recommendationTimer

async function get(path) {
  const response = await fetch(`${API}/${path}`)
  const payload = await response.json()
  if (!response.ok) throw new Error(payload.detail || 'Request failed')
  return payload
}

async function post(path, body) {
  const response = await fetch(`${API}/${path}`, {
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
  const response = await fetch(`${API}/extract-pdfs`, { method: 'POST', body: formData })
  const payload = await response.json()
  if (!response.ok) throw new Error(payload.detail || 'PDF extraction failed')
  return payload
}

function stats(items) {
  return items.map(([value, label]) => `<div><strong>${esc(value)}</strong><span>${esc(label)}</span></div>`).join('')
}

function steps(items) {
  return items.map((text, index) => `<article><b>${index + 1}</b><p>${esc(text)}</p></article>`).join('')
}

function vectorStats(vector) {
  return `
    <div class="vector-stats">
      ${stats([
        [vector.dimensions, 'dimensions'],
        [vector.magnitude, 'magnitude'],
        [vector.minimum, 'minimum value'],
        [vector.maximum, 'maximum value'],
        [vector.non_zero_values, 'non-zero values'],
      ])}
    </div>
    <code>[${vector.first_16_values.join(', ')} ...]</code>
  `
}

function fileCard(file) {
  const filename = file.path.split('/').pop()
  return `
    <article class="file-card">
      <strong>${esc(filename)}</strong>
      <span>${file.size_bytes} bytes</span>
      <p>${esc(file.stores)}</p>
      ${file.does_not_store ? `<p class="warning">Does not store: ${esc(file.does_not_store)}</p>` : ''}
      ${file.security_note ? `<p class="warning">${esc(file.security_note)}</p>` : ''}
      <details class="file-details">
        <summary>Show ${esc(filename)} details</summary>
        <pre>${esc(JSON.stringify(file.details, null, 2))}</pre>
      </details>
    </article>
  `
}

function currentOperation() {
  return document.querySelector('input[name="indexOperation"]:checked').value
}

function currentIndexRequest() {
  return {
    operation: currentOperation(),
    documents: sourceDocuments,
    text: sourceDocuments.length ? null : el('document').value,
    chunk_size: Number(el('chunkSize').value),
    chunk_overlap: Number(el('chunkOverlap').value),
  }
}

function renderRegistry(status) {
  if (!status.has_existing_vector_database) {
    el('registry').innerHTML = '<strong>Document registry:</strong> no saved vector database yet.'
    return
  }
  const documents = status.documents || []
  el('registry').innerHTML = `
    <div class="registry-heading"><strong>Saved document registry</strong><span>${documents.length} document(s) · last updated ${esc(status.updated_at || 'unknown')}</span></div>
    ${documents.length ? `<div class="table-wrap"><table><thead><tr><th>Filename</th><th>SHA-256 hash</th><th>Chunks</th><th>Indexed at</th></tr></thead><tbody>
      ${documents.map((document) => `<tr><td>${esc(document.filename)}</td><td><code>${esc(document.file_hash)}</code></td><td>${document.chunk_count}</td><td>${esc(document.indexed_at)}</td></tr>`).join('')}
    </tbody></table></div>` : '<p>This is a legacy index with no document registry.</p>'}
  `
}

async function refreshRecommendation() {
  try {
    const status = await get('index/status')
    renderRegistry(status)
    const request = currentIndexRequest()
    const recommendation = await post('index/recommendation', {
      documents: request.documents,
      text: request.text,
      chunk_size: request.chunk_size,
      chunk_overlap: request.chunk_overlap,
    })
    el('recommendation').className = `recommendation recommendation-${recommendation.recommended_operation}`
    el('recommendation').innerHTML = `<strong>Suggested: ${esc(recommendation.recommended_operation.replace('_', ' '))}</strong><span>${esc(recommendation.reason)}</span>`
  } catch (error) {
    el('recommendation').className = 'recommendation recommendation-error'
    el('recommendation').textContent = `Could not inspect the saved index: ${error.message}`
  }
}

function scheduleRecommendation() {
  clearTimeout(recommendationTimer)
  recommendationTimer = setTimeout(refreshRecommendation, 350)
}

function renderIndex(payload) {
  el('recommendation').className = 'recommendation recommendation-complete'
  el('recommendation').innerHTML = `<strong>${esc(payload.operation.message)}</strong>${payload.operation.skipped_duplicates.length ? `<span>Skipped duplicate hashes: ${esc(payload.operation.skipped_duplicates.join(', '))}</span>` : ''}`
  renderRegistry(payload.registry)
  el('modelSummary').innerHTML = stats([
    [payload.model.name, 'embedding model'],
    [payload.model.dimensions, 'vector dimensions'],
    [payload.model.max_sequence_length, 'maximum model tokens'],
    [payload.splitter.chunk_count, 'created chunks'],
    [payload.faiss.index_class, 'FAISS index class'],
    [payload.faiss.metric_type, 'distance metric'],
  ])
  el('indexingSteps').innerHTML = steps(payload.indexing_steps)
  el('splitterSummary').innerHTML = stats([
    [payload.splitter.type, 'splitter'],
    [payload.splitter.chunk_size_characters, 'maximum characters per chunk'],
    [payload.splitter.chunk_overlap_characters, 'requested overlap characters'],
    [payload.splitter.chunk_count, 'created chunks'],
  ])
  el('separators').innerHTML = payload.splitter.separators_tried_in_order.map((separator, index) => {
    const readable = separator === '\n\n' ? 'paragraph break \\n\\n'
      : separator === '\n' ? 'line break \\n'
        : separator === ' ' ? 'space'
          : 'individual characters'
    return `<span><b>${index + 1}</b>${readable}</span>`
  }).join('')
  el('splitterExplanation').textContent = payload.splitter.how_it_works
  el('chunkingRecords').innerHTML = payload.records.map((record, index) => `
    <article class="chunking-card">
      <div class="chunk-range">
        <strong>Chunk ${index}</strong>
        <span>characters ${record.chunking.start_character}-${record.chunking.end_character}</span>
        <span>${record.chunking.character_count} characters</span>
      </div>
      ${record.chunking.overlap_character_count
        ? `<div class="overlap"><strong>Repeated from previous chunk (${record.chunking.overlap_character_count} characters):</strong><code>${esc(record.chunking.overlap_text)}</code></div>`
        : '<div class="overlap no-overlap">First chunk has no previous overlap.</div>'}
      <p>${esc(record.text)}</p>
    </article>
  `).join('')
  el('records').innerHTML = payload.records.map((record) => `
    <article class="record">
      <div class="record-title"><strong>FAISS position ${record.faiss_position}</strong><code>${esc(record.document_id)}</code></div>
      <p>${esc(record.text)}</p>
      <div class="metadata">Metadata: <code>${esc(JSON.stringify(record.metadata))}</code></div>
      <details><summary>Tokenizer output: ${record.tokenization.tokens.length} tokens</summary>
        <div class="tokens">${record.tokenization.tokens.map((token, i) => `<span title="ID ${record.tokenization.token_ids[i]}, attention ${record.tokenization.attention_mask[i]}">${esc(token)}</span>`).join('')}</div>
        <p class="note">Special tokens such as <code>[CLS]</code> and <code>[SEP]</code> help the encoder structure its input.</p>
      </details>
      <details><summary>384-dimensional semantic embedding</summary>
        ${vectorStats(record.vector)}
        <p class="note">Individual dimensions do not correspond to named words. Meaning is distributed across the full vector.</p>
      </details>
      <div class="coordinate">2D teaching projection: (${record.pca_2d.join(', ')})</div>
    </article>
  `).join('')

  const files = payload.saved_files
  el('files').innerHTML = [files.index_faiss, files.index_pkl, files.registry_json].map(fileCard).join('')
  el('mapping').innerHTML = payload.records.map((record) => `
    <tr><td>${record.faiss_position}</td><td><code>${esc(record.document_id)}</code></td><td><code>${esc(JSON.stringify(record.metadata))}</code></td><td>${esc(record.text)}</td></tr>
  `).join('')
  el('indexResults').classList.remove('hidden')
  el('searchResults').classList.add('hidden')
}

function renderSearch(payload) {
  el('querySummary').innerHTML = stats([
    [payload.tokenization.tokens.length, 'question tokens'],
    [payload.query_vector.dimensions, 'embedding dimensions'],
    [payload.query_vector.magnitude, 'query magnitude'],
    [payload.nearest_neighbors.length, 'nearest neighbors returned'],
    [payload.results.length, 'neighbors accepted as relevant'],
  ])
  el('queryTokens').innerHTML = payload.tokenization.tokens.map((token, i) => `<span title="Token ID ${payload.tokenization.token_ids[i]}">${esc(token)}</span>`).join('')
  el('retrievalSteps').innerHTML = steps(payload.retrieval_steps)
  el('retrievalDecision').className = `decision ${payload.has_relevant_context ? 'decision-accept' : 'decision-reject'}`
  el('retrievalDecision').innerHTML = `
    <strong>${payload.has_relevant_context ? 'Relevant context accepted' : 'No relevant context found'}</strong>
    <p>${esc(payload.retrieval_decision)}</p>
    <small>FAISS always returns nearest vectors. The distance threshold decides whether they are relevant enough for an LLM.</small>
  `
  el('results').innerHTML = payload.nearest_neighbors.map((result) => `
    <article class="${result.l2_distance <= payload.maximum_l2_distance ? 'accepted-result' : 'rejected-result'}">
      <div class="rank">Rank ${result.rank}</div>
      <h3>FAISS position ${result.faiss_position}</h3>
      <div class="distance">Squared L2 distance: <strong>${result.l2_distance}</strong></div>
      <code>Manual check Σ(queryᵢ - chunkᵢ)² = ${result.manual_squared_l2}</code>
      <p>${esc(result.text)}</p>
      <div class="result-decision">${result.l2_distance <= payload.maximum_l2_distance ? 'Accepted as relevant' : 'Rejected as too distant'}</div>
      <small>LangChain document ID: ${esc(result.document_id)}</small>
    </article>
  `).join('')
  el('context').textContent = payload.context_for_llm || 'No context sent to the LLM. Safe answer: I don’t know based on the indexed document.'
  el('searchResults').classList.remove('hidden')
}

el('indexButton').addEventListener('click', async () => {
  el('indexError').textContent = ''
  try {
    renderIndex(await post('index', currentIndexRequest()))
  } catch (error) { el('indexError').textContent = error.message }
})

const defaultDocumentText = el('document').value

el('defaultTextButton').addEventListener('click', () => {
  sourceDocuments = []
  el('document').value = defaultDocumentText
  el('sourceStatus').textContent = 'Current source: default example text'
  el('pdfSummary').classList.add('hidden')
  el('indexError').textContent = ''
  refreshRecommendation()
})

el('document').addEventListener('input', () => {
  sourceDocuments = []
  el('sourceStatus').textContent = 'Current source: typed, pasted, or edited text'
  scheduleRecommendation()
})

el('uploadPdfButton').addEventListener('click', async () => {
  el('indexError').textContent = ''
  const files = Array.from(el('pdfFiles').files)
  if (!files.length) {
    el('indexError').textContent = 'Choose at least one PDF before extracting text.'
    return
  }
  try {
    el('sourceStatus').textContent = `Extracting text from ${files.length} PDF${files.length === 1 ? '' : 's'}...`
    const payload = await uploadPdfs(files)
    sourceDocuments = payload.files.map((file) => ({
      filename: file.filename,
      text: file.text,
      file_hash: file.file_hash,
    }))
    el('document').value = payload.combined_text
    el('sourceStatus').textContent = `Current source: ${payload.file_count} PDF${payload.file_count === 1 ? '' : 's'} · ${payload.total_pages_with_text}/${payload.total_pages} pages contained selectable text`
    el('pdfSummary').innerHTML = payload.files.map((file) => `
      <div><strong>${esc(file.filename)}</strong><span>${file.pages_with_text}/${file.page_count} pages with text</span></div>
    `).join('')
    el('pdfSummary').classList.remove('hidden')
    refreshRecommendation()
  } catch (error) {
    el('sourceStatus').textContent = 'PDF extraction failed'
    el('indexError').textContent = error.message
  }
})

el('searchButton').addEventListener('click', async () => {
  el('searchError').textContent = ''
  try {
    renderSearch(await post('search', {
      question: el('question').value,
      top_k: Number(el('topK').value),
      maximum_l2_distance: Number(el('maximumDistance').value),
    }))
  } catch (error) { el('searchError').textContent = error.message }
})

document.querySelectorAll('input[name="indexOperation"]').forEach((input) => input.addEventListener('change', refreshRecommendation))
el('chunkSize').addEventListener('change', refreshRecommendation)
el('chunkOverlap').addEventListener('change', refreshRecommendation)
refreshRecommendation()
