import { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import { adaptResult } from '../adapt'

const API_BASE_URL = 'http://localhost:8000/api'
const WS_URL = 'ws://localhost:8000/ws/logs'

// Етапи графа лекції (backend/lecture_graph.py)
const STAGES = [
  { id: 'check', label: 'Перевірка обсягу' },
  { id: 'analyst', label: 'Аналітик: блоки й довідка' },
  { id: 'allocate', label: 'Розподіл завдань і рівнів' },
  { id: 'generate', label: 'Генератор' },
  { id: 'verify', label: 'Критик і доопрацювання' },
  { id: 'done', label: 'Готово' },
]

const AGENT_COLORS = { System: '#666', Analyst: '#2196F3', Generator: '#4CAF50', Critic: '#FF9800', Error: '#F44336' }
const AGENT_NAMES = { System: 'Система', Analyst: 'Аналітик', Generator: 'Генератор', Critic: 'Критик', Error: 'Помилка' }

function LiveTerminalScreen({ lecture, onComplete }) {
  const navigate = useNavigate()
  const [logs, setLogs] = useState([])
  const [stage, setStage] = useState(null)
  const [counts, setCounts] = useState({ approved: 0, rejected: 0 })
  const [isConnected, setIsConnected] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState('')
  const logsEndRef = useRef(null)
  const startedRef = useRef(false)
  const finishedRef = useRef(false)
  const onCompleteRef = useRef(onComplete)
  useEffect(() => { onCompleteRef.current = onComplete }, [onComplete])

  useEffect(() => { logsEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [logs])

  const addLog = useCallback((agent, message, iteration = null) => {
    setLogs(prev => [...prev, { agent, message, iteration, time: new Date().toLocaleTimeString() }])
  }, [])

  const finish = useCallback((result) => {
    if (finishedRef.current) return
    finishedRef.current = true
    setIsGenerating(false); setStage('done')
    onCompleteRef.current(adaptResult(result))
    setTimeout(() => navigate('/results'), 800)
  }, [navigate])

  const start = useCallback(async () => {
    setIsGenerating(true); setError(''); finishedRef.current = false
    try {
      const response = lecture.mode === 'quick'
        ? await axios.post(`${API_BASE_URL}/generate-test`, {
            source_text: lecture.text, question_count: lecture.questionCount, configuration: 'M2',
          })
        : await axios.post(`${API_BASE_URL}/generate-lecture`, {
            text: lecture.text, question_count: lecture.questionCount, filename: lecture.filename,
          })
      if (response.data.success) finish(response.data.result)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
      setIsGenerating(false)
    }
  }, [lecture, finish])

  useEffect(() => {
    const ws = new WebSocket(WS_URL)
    ws.onopen = () => setIsConnected(true)
    ws.onclose = () => setIsConnected(false)
    ws.onerror = () => setError('Немає з’єднання з сервером (порт 8000)')
    ws.onmessage = (event) => {
      try {
        const d = JSON.parse(event.data)
        if (d.type === 'log') {
          if (d.stage) setStage(d.stage)
          if (d.verdict === 'APPROVED') setCounts(c => ({ ...c, approved: c.approved + 1 }))
          if (d.verdict === 'REJECT_FOR_REFINEMENT') setCounts(c => ({ ...c, rejected: c.rejected + 1 }))
          addLog(d.agent, d.message, d.iteration)
        } else if (d.type === 'complete') {
          finish(d.result)
        } else if (d.type === 'error') {
          addLog('Error', d.message); setError(d.message); setIsGenerating(false)
        }
      } catch (e) { console.error(e) }
    }
    return () => ws.close()
  }, [addLog, finish])

  useEffect(() => {
    if (isConnected && lecture.text && !startedRef.current) { startedRef.current = true; start() }
  }, [isConnected, lecture.text, start])

  if (!lecture.text) {
    return (
      <div className="terminal-screen"><div className="container">
        <p>Немає тексту лекції.</p>
        <button onClick={() => navigate('/')} className="btn btn-primary">На головну</button>
      </div></div>
    )
  }

  const quick = lecture.mode === 'quick'
  const stages = quick ? STAGES.filter(s => ['generate', 'verify', 'done'].includes(s.id)) : STAGES
  const stageIndex = stages.findIndex(s => s.id === stage)

  return (
    <div className="terminal-screen">
      <div className="container">
        <div className="terminal-header">
          <h1>Хід генерації</h1>
          <div className="connection-status">
            <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`} />
            {isConnected ? 'Підключено' : 'Відключено'}
          </div>
        </div>

        <div className="stage-tracker">
          {stages.map((s, i) => (
            <div key={s.id} className={`stage ${i < stageIndex || stage === 'done' ? 'done' : i === stageIndex ? 'active' : ''}`}>
              <span className="stage-dot">{i < stageIndex || stage === 'done' ? '✓' : i + 1}</span>
              <span className="stage-label">{s.label}</span>
            </div>
          ))}
        </div>

        <div className="terminal-content">
          <div className="source-text-section">
            <h3>Лекція</h3>
            <div className="source-text-preview">{lecture.text.substring(0, 400)}{lecture.text.length > 400 && '…'}</div>
            <p className="source-meta">
              {lecture.text.length.toLocaleString()} символів{lecture.filename ? ` · ${lecture.filename}` : ''} · питань: {lecture.questionCount} · {quick ? 'швидкий режим' : 'повний метод'}
              {' · '}вердиктів критика: схвалено {counts.approved}, відхилено {counts.rejected}
            </p>
          </div>

          <div className="logs-section">
            <h3>Журнал агентів</h3>
            <div className="logs-container">
              {logs.length === 0 && <div className="empty-logs"><p>Очікування подій…</p></div>}
              {logs.map((log, i) => (
                <div key={i} className="log-entry">
                  <span className="log-timestamp">{log.time}</span>
                  <span className="log-agent" style={{ color: AGENT_COLORS[log.agent] || '#666' }}>[{AGENT_NAMES[log.agent] || log.agent}]</span>
                  {log.iteration && <span className="log-iteration">ітерація {log.iteration}</span>}
                  <span className="log-message">{log.message}</span>
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </div>

          {error && <div className="error-section"><h3>Помилка</h3><p className="error-message">{error}</p></div>}

          <div className="terminal-actions">
            {!isGenerating && (
              <button onClick={start} disabled={!isConnected} className="btn btn-primary">Запустити ще раз</button>
            )}
            <button onClick={() => navigate('/')} className="btn btn-secondary">Назад</button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default LiveTerminalScreen
