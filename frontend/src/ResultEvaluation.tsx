import { short } from './api'
import { ExecutionStatus } from './ResultMatrix'
import { ReadableValue } from './components/StructuredValue'
import { comparisonKey, ranking } from './resultComparison'
type Row = Record<string, any>
export default function ResultEvaluation({
  groupId,
  values,
  comparison,
  onChange,
  openRun,
}: {
  groupId: string
  values: Row[]
  comparison: Row
  onChange: (value: Row) => void
  openRun: (id: string) => void
}) {
  const name = values[0].definition.name
  const metrics = [
    ...new Set(values.flatMap((v) => v.summaries.flatMap((s: Row) => Object.keys(s.metrics)))),
  ] as string[]
  const metric = comparison.metrics?.[groupId] || metrics[0] || 'accuracy'
  const direction =
    comparison.directions?.[groupId] ||
    (metric === 'accuracy' || metric.endsWith('/accuracy') ? 'higher' : '')
  const mismatched = new Set(values.map(comparisonKey)).size > 1
  const entries = values.flatMap((v) =>
    v.summaries.map((s: Row) => ({ v, s, key: comparisonKey(v), score: s.metrics[metric] })),
  )
  const comparable = (key: string) =>
    entries
      .filter(
        (e) =>
          (comparison.allow_mismatch || e.key === key) &&
          e.s.scored === e.s.total &&
          typeof e.score === 'number',
      )
      .map((e) => e.score)
  const update = (field: string, value: string) =>
    onChange({ ...comparison, [field]: { ...comparison[field], [groupId]: value } })
  return (
    <details className="panel result-breakdown">
      <summary>{name}</summary>
      {mismatched && (
        <p role="alert" className="alert">
          {comparison.allow_mismatch
            ? 'Override enabled: rankings compare different inputs or scoring settings.'
            : 'These runs differ in revision, sample, repeats, scoring or metrics. Rankings are separate within matching groups.'}
        </p>
      )}
      <div className="form-columns">
        <label>
          Metric for {name}
          <select value={metric} onChange={(e) => update('metrics', e.target.value)}>
            {(metrics.length ? metrics : ['accuracy']).map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </label>
        <label>
          Better values for {name}
          <select value={direction} onChange={(e) => update('directions', e.target.value)}>
            <option value="">Choose direction to enable ranking</option>
            <option value="higher">Higher is better</option>
            <option value="lower">Lower is better</option>
          </select>
        </label>
      </div>
      <table>
        <caption>
          Bold: best; underline: second-best distinct value within comparable, fully scored entries.
        </caption>
        <thead>
          <tr>
            <th>Model / run</th>
            <th>{metric}</th>
            <th>Scored</th>
            <th>Inspector assessed</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(({ v, s, key, score }) => {
            const order = ranking(comparable(key), direction)
            const rank = direction && s.scored === s.total ? order.indexOf(score) : -1
            const text = typeof score === 'number' ? score.toFixed(4) : 'Pending'
            return (
              <tr key={v.run.id + s.model}>
                <td>
                  {s.model}
                  <br />
                  <button className="link" onClick={() => openRun(v.run.id)}>
                    {short(v.run.id)} · revision {short(v.run.revision_id)}
                  </button>
                </td>
                <td>{rank === 0 ? <strong>{text}</strong> : rank === 1 ? <u>{text}</u> : text}</td>
                <td>
                  {s.scored}/{s.total}
                </td>
                <td>
                  {s.inspector_assessed}/{s.total}
                </td>
                <td>
                  <ExecutionStatus summary={s} status={v.run.status} />
                  {s.metric_errors.map((e: string) => (
                    <p role="alert" key={e}>
                      {e}
                    </p>
                  ))}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <h3>{metric} by model and run</h3>
      {entries.map(({ v, s, score }) => (
        <div key={v.run.id + s.model}>
          <span>
            {s.model} · {short(v.run.id)}:{' '}
            {typeof score === 'number' ? score.toFixed(4) : 'Pending'}
          </span>
          {typeof score === 'number' && (
            <meter
              style={{ width: '100%' }}
              aria-label={`${s.model} ${short(v.run.id)} ${metric}`}
              min={Math.min(
                0,
                ...entries.filter((e) => typeof e.score === 'number').map((e) => e.score),
              )}
              max={Math.max(
                1,
                ...entries.filter((e) => typeof e.score === 'number').map((e) => e.score),
              )}
              value={score}
            />
          )}
        </div>
      ))}
      <h3>Response outcomes</h3>
      {entries.map(({ v, s }) => {
        const correct = s.scored_correct
        const incorrect = s.scored - correct,
          unscored = s.total - s.scored
        return (
          <div key={v.run.id + s.model}>
            <p>
              {s.model} · {short(v.run.id)}: {correct} scored correct · {incorrect} scored incorrect
              · {unscored} unscored
            </p>
            <div
              className="outcome-stack"
              role="img"
              aria-label={`${correct} scored correct, ${incorrect} scored incorrect, ${unscored} unscored`}
            >
              <span style={{ width: `${(100 * correct) / s.total}%`, background: '#5d8c4d' }} />
              <span style={{ width: `${(100 * incorrect) / s.total}%`, background: '#bf745a' }} />
              <span style={{ width: `${(100 * unscored) / s.total}%`, background: '#b9bdc2' }} />
            </div>
          </div>
        )
      })}
      <details>
        <summary>Compare inputs and scoring settings</summary>
        {values.map((v) => (
          <section key={v.run.id}>
            <h4>Run {short(v.run.id)}</h4>
            <ReadableValue
              value={{
                revision: v.run.revision_id,
                sample_ids: v.run.manifest.ids,
                repeats: v.run.config.repeats,
                scorer: v.definition.scorer,
                metrics: v.definition.metrics,
              }}
            />
          </section>
        ))}
      </details>
    </details>
  )
}
