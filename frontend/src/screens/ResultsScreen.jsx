import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

// Статус завдання після циклу критика (backend/lecture_graph.py)
function statusOf(s) {
  if (s === 'passed_at_iter_1') return { text: 'Схвалено з першої ітерації', cls: 'ok' }
  if (s?.startsWith('fixed_at_iter_')) return { text: `Виправлено на ${s.split('_').pop()}-й ітерації`, cls: 'fixed' }
  if (s === 'unresolved_after_cap') return { text: 'Не виправлено в межах ліміту ітерацій', cls: 'bad' }
  if (s === 'hard_reject') return { text: 'Відхилено (виправлення неможливе)', cls: 'bad' }
  if (s === 'm3_rejected_by_critic') return { text: 'Відхилено критиком', cls: 'bad' }
  if (s === 'quick') return { text: 'Швидкий режим', cls: 'plain' }
  return { text: s || 'не завершено', cls: 'bad' }
}

const ELEM = { question: 'основа', key: 'ключ', d1: 'дистрактор 1', d2: 'дистрактор 2', d3: 'дистрактор 3', options: 'усі варіанти' }
const STAGE = { 1: 'етап 1 (фактологія, однозначність)', 2: 'етап 2 (правдоподібність, підказки)' }
const BLOCK_TYPE = { content: 'змістовий', intro: 'вступ', summary: 'підсумок', organizational: 'організаційний', repeat: 'повтор' }
const LETTERS = ['A', 'B', 'C', 'D']

// Стабільне перемішування варіантів: ключ не завжди на першому місці, але порядок однаковий після перезавантаження
function hash(str) { let h = 2166136261; for (const c of str) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619) } return h >>> 0 }
function arrange(mcq, seed) {
  const opts = [{ text: mcq.key, isKey: true }, ...mcq.distractors.map(d => ({ text: d.text, note: d.misconception_modeled }))]
  let h = hash(seed)
  for (let i = opts.length - 1; i > 0; i--) { h = Math.imul(h ^ (h >>> 15), 2246822507) >>> 0; const j = h % (i + 1); [opts[i], opts[j]] = [opts[j], opts[i]] }
  return opts
}

function changed(prev, cur) {
  if (!prev) return []
  const out = []
  if (prev.question !== cur.question) out.push('question')
  if (prev.key !== cur.key) out.push('key')
  cur.distractors.forEach((d, i) => { if (prev.distractors[i]?.text !== d.text) out.push(`d${i + 1}`) })
  return out
}

const stageNo = (s) => (s?.startsWith('Stage_1') ? 1 : s?.startsWith('Stage_2') ? 2 : null)

function Iteration({ it, prev }) {
  const rejected = it.verdict === 'REJECT_FOR_REFINEMENT'
  const ch = changed(prev, it.mcq)
  return (
    <div className={`iter-box ${rejected ? 'rejected' : 'approved'}`}>
      <div className="iter-head">
        <span className="iter-title">Ітерація {it.iter}</span>
        <span className={`iter-badge ${rejected ? 'rejected' : 'approved'}`}>{rejected ? 'Відхилено' : 'Схвалено'}</span>
        {rejected && stageNo(it.failed_stage) && <span className="iter-note">{STAGE[stageNo(it.failed_stage)]}</span>}
        {prev && <span className="iter-note">змінено: {ch.length ? ch.map(e => ELEM[e]).join(', ') : 'нічого'}</span>}
      </div>
      {rejected && (
        <div className="iter-report">
          <p><b>Дефектні елементи:</b> {it.targets.map(e => ELEM[e] || e).join(', ') || '—'}</p>
          <p><b>Причина дефекту:</b> {it.defect_explanation}</p>
          <p><b>Вказівка щодо виправлення:</b> {it.instruction}</p>
        </div>
      )}
      <details open={rejected}>
        <summary>Версія завдання на цій ітерації</summary>
        <p className="iter-q">{it.mcq.question}</p>
        <ul>
          <li><b>Ключ:</b> {it.mcq.key}</li>
          {it.mcq.distractors.map((d, k) => <li key={k}>{d.text}</li>)}
        </ul>
      </details>
    </div>
  )
}

function QuestionCard({ item, n }) {
  const [open, setOpen] = useState(false)
  const st = statusOf(item.status)
  const opts = arrange(item.final, item.id)
  const its = item.iterations || []
  const cleanPass = its.length === 1 && its[0].verdict === 'APPROVED'
  return (
    <div className={`q-card ${st.cls}`}>
      <div className="q-head">
        <span className="q-num">Питання {n}</span>
        {(item.block_title || item.target_level) && (
          <span className="q-meta">
            {item.block_title ? `блок «${item.block_title}»` : ''}
            {item.block_title && item.target_level ? ' · ' : ''}
            {item.target_level ? `цільовий рівень Блума ${item.target_level}` : ''}
          </span>
        )}
        <span className={`q-status ${st.cls}`}>{st.text}</span>
      </div>
      <p className="q-text">{item.final.question}</p>
      <div className="q-options">
        {opts.map((o, i) => (
          <div key={i} className={`q-option ${o.isKey ? 'key' : ''}`}>
            <div><span className="q-letter">{LETTERS[i]})</span> {o.text}{o.isKey && <span className="q-key"> ✓ правильна відповідь</span>}</div>
            {o.note && <div className="q-note">Типова помилка студента, яку моделює варіант: {o.note}</div>}
          </div>
        ))}
      </div>
      {cleanPass && <p className="q-clean">Критик схвалив завдання без зауважень.</p>}
      {!cleanPass && its.length > 0 && its.some(it => it.verdict) && (
        <>
          <button className="btn-history" onClick={() => setOpen(!open)}>
            {open ? 'Сховати' : 'Показати'} історію доопрацювання ({its.length} {its.length === 1 ? 'ітерація' : its.length < 5 ? 'ітерації' : 'ітерацій'})
          </button>
          {open && its.map((it, k) => <Iteration key={it.iter} it={it} prev={k > 0 ? its[k - 1].mcq : null} />)}
        </>
      )}
      {item.fail && <p className="q-fail">Генератор: {item.fail}</p>}
    </div>
  )
}

function ResultsScreen({ result, onNewTest }) {
  const navigate = useNavigate()
  const [showBlocks, setShowBlocks] = useState(false)

  if (!result) {
    return (
      <div className="results-screen"><div className="container">
        <h1>Немає результатів</h1>
        <button onClick={() => navigate('/')} className="btn btn-primary">Нова лекція</button>
      </div></div>
    )
  }

  const items = result.items || []
  const counts = {}
  items.forEach(it => { const t = statusOf(it.status); counts[t.text] = counts[t.text] || { n: 0, cls: t.cls }; counts[t.text].n++ })
  const order = (k) => (k.startsWith('Схвалено') ? 0 : k.startsWith('Виправлено') ? 1 + parseInt(k.match(/\d+/)?.[0] || 0, 10) : 99)
  const hasBad = items.some(it => statusOf(it.status).cls === 'bad')

  const download = (data, name, type) => {
    const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([data], { type })); a.download = name; a.click()
  }
  const exportJSON = () => download(JSON.stringify(result, null, 2), `test_${result.run_id || 'lecture'}.json`, 'application/json')
  const exportTXT = () => {
    const lines = []
    items.forEach((it, i) => {
      const opts = arrange(it.final, it.id)
      lines.push(`${i + 1}. ${it.final.question}`)
      opts.forEach((o, k) => lines.push(`   ${LETTERS[k]}) ${o.text}`))
      lines.push(`   Відповідь: ${LETTERS[opts.findIndex(o => o.isKey)]}`, '')
    })
    download(lines.join('\n'), `test_${result.run_id || 'lecture'}.txt`, 'text/plain;charset=utf-8')
  }

  return (
    <div className="results-screen">
      <div className="container">
        <div className="results-header">
          <h1>{result.quick ? 'Питання з фрагмента' : 'Набір питань за лекцією'}</h1>
          <div className="pills">
            {result.domain && <span className="pill">Галузь: {result.domain}</span>}
            {!result.quick && <span className="pill">Блоків: {result.blocks?.length || 0}</span>}
            <span className="pill">Питань: {items.length}</span>
          </div>
        </div>

        <div className="status-chips">
          {Object.entries(counts).sort((a, b) => order(a[0]) - order(b[0])).map(([k, v]) => (
            <span key={k} className={`chip ${v.cls}`}>{k}: {v.n}</span>
          ))}
        </div>
        {hasBad && (
          <p className="results-note">
            Питання зі статусом «не виправлено» або «відхилено» Критик не зміг довести до схвалення: їх варто переглянути вручну перед використанням.
          </p>
        )}

        <div className="results-actions">
          {!result.quick && (
            <button onClick={() => setShowBlocks(!showBlocks)} className="btn btn-secondary">
              {showBlocks ? 'Сховати поділ лекції на блоки' : 'Показати поділ лекції на блоки'}
            </button>
          )}
          <button onClick={exportJSON} className="btn btn-secondary">Експорт JSON</button>
          <button onClick={exportTXT} className="btn btn-secondary">Експорт тесту (TXT з ключами)</button>
          <button onClick={() => { onNewTest(); navigate('/') }} className="btn btn-primary">Нова лекція</button>
        </div>

        {showBlocks && (
          <div className="blocks-panel">
            <table>
              <thead><tr><th>№</th><th>Блок</th><th>Тип</th><th>Стеля рівня Блума</th><th>Питань</th></tr></thead>
              <tbody>
                {result.blocks.map(b => (
                  <tr key={b.id}>
                    <td>{b.id}</td><td>{b.title}</td><td>{BLOCK_TYPE[b.type] || b.type}</td>
                    <td>{b.bloom || '—'}</td><td>{items.filter(it => it.block === b.id).length}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {items.map((it, i) => <QuestionCard key={it.id} item={it} n={i + 1} />)}
      </div>
    </div>
  )
}

export default ResultsScreen
