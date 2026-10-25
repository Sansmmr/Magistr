import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000/api'
const MAX_CHARS = 50000          // rules.json -> limits.lecture_max_chars
const MIN_CHARS = 200
const MIN_Q = 3
const MAX_Q = 30

function UploadScreen({ onStart }) {
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState('file')
  const [text, setText] = useState('')
  const [file, setFile] = useState(null)
  const [filename, setFilename] = useState(null)
  const [questionCount, setQuestionCount] = useState(12)
  const [isLoading, setIsLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState('')

  const length = text.trim().length
  const tooLong = length > MAX_CHARS
  const tooShort = length < MIN_CHARS
  const countOk = Number.isInteger(questionCount) && questionCount >= MIN_Q && questionCount <= MAX_Q

  const pickFile = (f) => { if (f) { setFile(f); setError('') } }

  const extractText = async () => {
    if (!file) return
    setIsLoading(true); setError('')
    try {
      const formData = new FormData()
      formData.append('file', file)
      const response = await axios.post(`${API_BASE_URL}/upload-file`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setText(response.data.text || '')
      setFilename(file.name)
    } catch (err) {
      setError('Не вдалося вилучити текст: ' + (err.response?.data?.detail || err.message))
    } finally {
      setIsLoading(false)
    }
  }

  const start = () => {
    if (tooShort) { setError(`Замало тексту: потрібно щонайменше ${MIN_CHARS} символів лекції.`); return }
    if (tooLong) { setError(`Лекція завелика: ${length.toLocaleString()} символів, допустимо не більше ${MAX_CHARS.toLocaleString()}. Розділіть її на частини за темами.`); return }
    if (!countOk) { setError(`Кількість питань — від ${MIN_Q} до ${MAX_Q}.`); return }
    onStart(text.trim(), questionCount, filename, 'full')
    navigate('/terminal')
  }

  const ready = !tooShort && !tooLong

  return (
    <div className="upload-screen">
      <div className="upload-card">
        <h1>Система генерації тестів</h1>
        <p className="subtitle">Генерація тестових питань на основі багатоагентного ШІ</p>

        <div className="method-tabs">
          <button className={`tab ${activeTab === 'file' ? 'active' : ''}`} onClick={() => setActiveTab('file')}>
            Завантаження файлу
          </button>
          <button className={`tab ${activeTab === 'text' ? 'active' : ''}`} onClick={() => setActiveTab('text')}>
            Введення тексту
          </button>
        </div>

        {activeTab === 'file' && (
          <div
            className={`drop-zone ${dragOver ? 'over' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); pickFile(e.dataTransfer.files[0]) }}
          >
            <p className="drop-help">
              Завантажте файл з <b>навчальним текстом</b> (лекція, стаття). Система створить <b>нові</b> питання
              на основі змісту файлу — вкладка «Введення тексту» не потрібна.
            </p>
            <input type="file" id="file-upload" accept=".pdf,.txt,.docx" onChange={(e) => pickFile(e.target.files[0])} className="file-input" />
            <label htmlFor="file-upload" className="drop-label">
              <span className="drop-icon">📄</span>
              <span className="drop-title">Натисніть, щоб вибрати файл, або перетягніть його сюди</span>
              <span className="drop-hint">PDF, TXT, DOCX</span>
            </label>
            {file && <p className="drop-file">{file.name}</p>}
            {file && (
              <button onClick={extractText} disabled={isLoading} className="btn btn-primary">
                {isLoading ? 'Обробка…' : 'Вилучити текст'}
              </button>
            )}
            {filename && length > 0 && (
              <div className={`ready-box ${ready ? '' : 'bad'}`}>
                <p className="ready-title">{ready ? '✓ Файл готовий до генерації' : '✗ Текст не підходить для генерації'}</p>
                <p>Файл: {filename}</p>
                <p>Текст: {length.toLocaleString()} символів{tooLong ? ` (більше ніж ${MAX_CHARS.toLocaleString()})` : ''}</p>
                <p className="ready-start">Початок: {text.trim().replace(/\s+/g, ' ').substring(0, 140)}…</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'text' && (
          <div className="text-input">
            <p className="upload-help">
              Вставте текст однієї лекції (до {MAX_CHARS.toLocaleString()} символів). Більшу лекцію розділіть на частини за темами.
            </p>
            <textarea
              value={text}
              onChange={(e) => { setText(e.target.value); setFilename(null) }}
              placeholder="Текст лекції…"
              className="text-area"
              rows={14}
            />
            <p className={`char-counter ${tooLong ? 'over' : ''}`}>
              {length.toLocaleString()} / {MAX_CHARS.toLocaleString()} символів
            </p>
          </div>
        )}

        <div className="settings-section">
          <h3>Налаштування генерації</h3>
          <div className="count-row">
            <label htmlFor="qcount">Кількість питань ({MIN_Q}–{MAX_Q}):</label>
            <input
              id="qcount" type="number" min={MIN_Q} max={MAX_Q} value={Number.isNaN(questionCount) ? '' : questionCount}
              onChange={(e) => setQuestionCount(parseInt(e.target.value, 10))}
              className="number-input"
            />
          </div>
        </div>

        {error && <div className="error-message">{error}</div>}

        <button onClick={start} disabled={isLoading || tooShort || tooLong || !countOk} className="btn btn-large btn-primary btn-block">
          Почати генерацію
        </button>
      </div>
    </div>
  )
}

export default UploadScreen
