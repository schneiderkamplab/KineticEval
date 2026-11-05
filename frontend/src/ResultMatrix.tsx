import { short } from './api'
type Row = Record<string, any>
import { canonical } from './canonical'
export const modelKey = (model: Row) => JSON.stringify(canonical(model))
export function ExecutionStatus({ summary: s, status }: { summary: Row; status: string }) {
  const labels: Record<string, string> = {
    generation: 'generating',
    scoring: 'scoring',
    diagnostic: 'checking heuristics',
    inspector: 'inspecting',
  }
  const stages = Object.entries(s.execution || {}) as [string, Row][]
  const running = stages
    .filter(([, v]) => v.running)
    .map(([stage, v]) => `${labels[stage]} ${v.running}`)
  const queued = stages
    .filter(([, v]) => v.pending)
    .map(([stage, v]) => `${labels[stage]} ${v.pending}`)
  const failed = stages.reduce((n, [, v]) => n + (v.failed || 0), 0)
  const text =
    failed && !running.length && !queued.length
      ? 'Failed · ' + failed + ' jobs'
      : status !== 'active'
        ? status.replaceAll('_', ' ')
        : running.length
          ? 'Running · ' + running.join(' · ')
          : s.metric_status === 'running'
            ? 'Running · computing metrics'
            : queued.length
              ? 'Queued · ' + queued.join(' · ')
              : s.metric_status === 'pending'
                ? 'Queued · computing metrics'
                : s.metric_status === 'succeeded'
                  ? 'Scores ready · awaiting remaining run analysis'
                  : 'Waiting for earlier stages'
  return (
    <>
      <small className={running.length ? 'execution-running' : 'execution-waiting'}>{text}</small>
      {s.execution_errors?.length > 0 && (
        <details>
          <summary>Failure details</summary>
          {s.execution_errors.map((error: string) => (
            <p key={error}>{error}</p>
          ))}
        </details>
      )}
    </>
  )
}
export function ResultMatrix({
  groups,
  comparison,
  comparisonKey,
  openRun,
}: {
  groups: Record<string, Row[]>
  comparison: Row
  comparisonKey: (v: Row) => string
  openRun: (id: string) => void
}) {
  const models = new Map<string, Row>()
  for (const values of Object.values(groups))
    for (const value of values)
      for (const model of value.run.config.models) models.set(modelKey(model), model)
  function cell(values: Row[], group: string, key: string) {
    const all = values.flatMap((v) =>
      v.summaries.map((s: Row) => ({
        v,
        s,
        model: v.run.config.models.find((m: Row) => m.name === s.model),
      })),
    )
    const metrics = [...new Set(all.flatMap((e) => Object.keys(e.s.metrics)))]
    const metric = comparison.metrics?.[group] || metrics[0] || 'accuracy'
    const direction =
      comparison.directions?.[group] ||
      (metric === 'accuracy' || metric.endsWith('/accuracy') ? 'higher' : '')
    const mine = all.filter((e) => e.model && modelKey(e.model) === key)
    if (!mine.length) return <span className="muted">—</span>
    return mine.map(({ v, s }) => {
      const score = s.metrics[metric]
      const candidates = all
        .filter(
          (e) =>
            (comparison.allow_mismatch || comparisonKey(e.v) === comparisonKey(v)) &&
            e.s.scored === e.s.total &&
            typeof e.s.metrics[metric] === 'number',
        )
        .map((e) => e.s.metrics[metric])
      const order = [...new Set<number>(candidates)].sort((a, b) =>
        direction === 'lower' ? a - b : b - a,
      )
      const rank = direction && s.scored === s.total ? order.indexOf(score) : -1
      const text =
        typeof score === 'number'
          ? score.toFixed(4)
          : Object.values(s.execution || {}).some((stage: any) => stage.failed)
            ? 'Unavailable'
            : 'Pending'
      return (
        <div key={v.run.id}>
          <span>{rank === 0 ? <strong>{text}</strong> : rank === 1 ? <u>{text}</u> : text}</span>
          <small>
            {metric} · {s.scored}/{s.total} scored
          </small>
          {typeof s.reused === 'number' && (
            <small>
              {s.reused}/{s.total} responses reused
            </small>
          )}
          {typeof score !== 'number' && <ExecutionStatus summary={s} status={v.run.status} />}
          <button className="link" onClick={() => openRun(v.run.id)}>
            Run {short(v.run.id)}
          </button>
        </div>
      )
    })
  }
  return (
    <section className="panel">
      <h2>Model × evaluation comparison</h2>
      <p>
        Bold marks the best; underline marks the second-best distinct value. Rankings use the
        comparison settings below. Repeated runs remain separate within each cell.
      </p>
      {Object.values(groups).some((values) => new Set(values.map(comparisonKey)).size > 1) && (
        <p role="alert">
          {comparison.allow_mismatch
            ? 'Mismatch override enabled: some rankings compare different inputs or scoring settings.'
            : 'Mismatched runs are ranked only within comparable groups.'}
        </p>
      )}
      <div
        className="comparison-scroll"
        tabIndex={0}
        role="region"
        aria-label="Scrollable model comparison"
      >
        <table className="comparison-matrix" aria-label="Model by evaluation comparison">
          <thead>
            <tr>
              <th scope="col">Model configuration</th>
              {Object.entries(groups).map(([id, values]) => (
                <th scope="col" key={id}>
                  {values[0].definition.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[...models].map(([key, model]) => (
              <tr key={key}>
                <th scope="row">
                  {model.name}
                  {model.model_identifier && model.model_identifier !== model.name && (
                    <small>{model.model_identifier}</small>
                  )}
                  <details>
                    <summary>Generation settings</summary>
                    <dl>
                      {Object.entries(model.parameters).map(([k, v]) => (
                        <div key={k}>
                          <dt>{k.replaceAll('_', ' ')}</dt>
                          <dd>{String(v)}</dd>
                        </div>
                      ))}
                    </dl>
                    {model.chat_template && (
                      <>
                        <p>Custom chat template</p>
                        <pre>{model.chat_template}</pre>
                      </>
                    )}
                  </details>
                </th>
                {Object.entries(groups).map(([id, values]) => (
                  <td key={id}>{cell(values, id, key)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
