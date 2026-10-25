// Приводить результат швидкого режиму (/api/generate-test, окремі питання з фрагмента)
// до формату лекційного конвеєра (/api/generate-lecture), щоб екран результатів був спільним.
export function adaptResult(result) {
  if (!result || result.items) return result
  const qs = result.questions || []
  const items = qs.filter(q => !q.error).map((q, i) => {
    const g = q.generated_question || {}
    const opts = g.options || {}
    const keyLetter = g.correct_answer || 'A'
    const mcq = {
      question: g.question_stem || '',
      key: opts[keyLetter] || '',
      distractors: ['A', 'B', 'C', 'D'].filter(l => l !== keyLetter && opts[l]).map((l, k) => ({ id: `d${k + 1}`, text: opts[l] })),
    }
    const v = q.verification || {}
    return {
      id: `q${i + 1}`, block: null, block_title: null, block_bloom: null, target_level: null,
      status: v.approved ? 'passed_at_iter_1' : 'quick',
      final: mcq,
      iterations: [{ iter: 1, mcq, verdict: v.approved ? 'APPROVED' : null, targets: [], defect_explanation: '', instruction: '' }],
      fail: null,
    }
  })
  return { summary: {}, domain: null, blocks: [], items, quick: true }
}
