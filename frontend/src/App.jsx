import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import './App.css'
import UploadScreen from './screens/UploadScreen'
import LiveTerminalScreen from './screens/LiveTerminalScreen'
import ResultsScreen from './screens/ResultsScreen'

// Вебзастосунок запускає повну конфігурацію методу (M4) для лекції користувача:
// аналітик -> розподіл завдань -> генератор -> критик із циклом доопрацювання (backend/lecture_graph.py).
function App() {
  const [lecture, setLecture] = useState({ text: '', filename: null, questionCount: 12, mode: 'full' })
  const [result, setResult] = useState(null)

  return (
    <Router>
      <div className="app">
        <Routes>
          <Route
            path="/"
            element={<UploadScreen onStart={(text, questionCount, filename, mode) => setLecture({ text, questionCount, filename, mode })} />}
          />
          <Route
            path="/terminal"
            element={<LiveTerminalScreen lecture={lecture} onComplete={(data) => setResult(data)} />}
          />
          <Route
            path="/results"
            element={<ResultsScreen result={result} onNewTest={() => { setResult(null); setLecture({ text: '', filename: null, questionCount: 12, mode: 'full' }) }} />}
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App
