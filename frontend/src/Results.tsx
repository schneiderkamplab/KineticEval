import RunProgress from './RunProgress'
import { ResultMatrix } from './ResultMatrix'
import { useEffect, useState } from 'react'
import { api, short } from './api'
import ResultCollectionDialog from './ResultCollectionDialog'
import { comparisonKey } from './resultComparison'
export { comparisonKey, ranking } from './resultComparison'
import ResultEvaluation from './ResultEvaluation'
import { useResultData } from './hooks/useResultData'
type Row = Record<string, any>

export default function Results({
  initialId,
  runs,
  onLaunch,
  openRun,
}: {
  initialId: string
  runs: Row[]
  onLaunch: () => void
  openRun: (id: string) => void
}) {
  const [selected, setSelected] = useState(initialId),
    [draft, setDraft] = useState<Row | null>(null),
    [error, setError] = useState(''),
    [busy, setBusy] = useState(false)
  const [retryMessage, setRetryMessage] = useState('')
  const { rows, data, error: loadError, refresh } = useResultData(selected)
  const current = rows.find((r) => r.id === selected)
  useEffect(() => {
    setSelected(initialId)
  }, [initialId])
  useEffect(() => {
    setRetryMessage('')
    setError('')
  }, [selected])
  async function retryFailed() {
    if (!current) return
    setBusy(true)
    setError('')
    setRetryMessage('')
    try {
      const result = await api('/results/' + current.id + '/retry', {})
      setRetryMessage(
        result.jobs_retried
          ? `Scheduled retries for ${result.jobs_retried} failed stages across ${result.runs_retried} runs.`
          : 'No failed stages to retry.',
      )
      refresh()
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }
  async function save(body: Row) {
    setBusy(true)
    setError('')
    try {
      const row = await api('/results' + (body.id ? '/' + body.id : ''), body)
      refresh()
      setSelected(row.id)
      setDraft(null)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }
  async function removeCollection() {
    if (!draft?.id) return
    setBusy(true)
    setError('')
    try {
      const response = await fetch('/api/results/' + draft.id, { method: 'DELETE' })
      if (!response.ok) throw new Error('Could not delete collection')
      setDraft(null)
      setSelected('')
      refresh()
    } catch (error) {
      setError(String(error))
    } finally {
      setBusy(false)
    }
  }
  const groups: Record<string, Row[]> = {}
  for (const value of data)
    (groups[value.run.evaluation_id || value.definition.name] ??= []).push(value)
  return (
    <>
      <div className="page-heading">
        <div>
          <h1>Results</h1>
          <p>Collections of runs, with explicit comparison settings and coverage.</p>
        </div>
        <button onClick={onLaunch}>Create runs</button>
      </div>
      {(error || loadError) && (
        <p role="alert" className="error">
          {error || loadError}
        </p>
      )}
      <div className="toolbar">
        <label>
          Result
          <select
            disabled={busy}
            aria-label="Result"
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
          >
            <option value="">Select a Result</option>
            {rows.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name} · {new Date(r.created * 1000).toLocaleString()}
              </option>
            ))}
          </select>
        </label>
        <button
          onClick={() =>
            setDraft({ name: 'Comparison', description: '', run_ids: [], comparison: {} })
          }
        >
          Group existing runs
        </button>
        {current && (
          <>
            <button disabled={busy} onClick={() => setDraft(structuredClone(current))}>
              Edit collection
            </button>
            <button disabled={busy} onClick={() => void retryFailed()}>
              Retry failed stages
            </button>
          </>
        )}
      </div>
      {retryMessage && <p role="status">{retryMessage}</p>}
      {current && (
        <>
          <h2>{current.name}</h2>
          <p>{current.description}</p>
          <p>
            Created {new Date(current.created * 1000).toLocaleString()} · {current.run_ids.length}{' '}
            runs
          </p>
          <label className="check">
            <input
              type="checkbox"
              checked={!!current.comparison.allow_mismatch}
              onChange={(e) =>
                void save({
                  ...current,
                  comparison: { ...current.comparison, allow_mismatch: e.target.checked },
                })
              }
            />{' '}
            Compare mismatched runs (explicit override)
          </label>
          {!current.run_ids.length && (
            <p>This Result has no runs. Add existing runs with Edit collection.</p>
          )}
        </>
      )}
      {current && data.some((v) => v.run.status === 'active') && (
        <section className="panel">
          <h2>Live run activity</h2>
          {data
            .filter((v) => v.run.status === 'active')
            .map((v) => (
              <section key={v.run.id}>
                <h3>
                  {v.definition.name} ·{' '}
                  <button className="link" onClick={() => openRun(v.run.id)}>
                    {short(v.run.id)}
                  </button>
                </h3>
                <RunProgress id={v.run.id} status={v.run.status} compact />
              </section>
            ))}
        </section>
      )}
      {current &&
        data.some((v) =>
          [v.report, ...(v.investigations || [])].some((a) => a?.status === 'failed'),
        ) && (
          <details className="panel result-breakdown">
            <summary>
              Analysis failures ·{' '}
              {data.reduce(
                (count, value) =>
                  count +
                  [value.report, ...(value.investigations || [])].filter(
                    (artifact) => artifact?.status === 'failed',
                  ).length,
                0,
              )}{' '}
              stages
            </summary>
            <p>
              These failures affect investigations or reports. Completed scores remain available.
            </p>
            {data.map((v) => {
              const failed = [
                { label: 'Report', artifact: v.report },
                ...(v.investigations || []).map((artifact: Row) => ({
                  label: 'Pattern investigation',
                  artifact,
                })),
              ].filter((a) => a.artifact?.status === 'failed')
              return (
                failed.length > 0 && (
                  <details key={v.run.id}>
                    <summary>
                      {v.definition.name} · {failed.length} failed analysis stages
                    </summary>
                    {failed.map((a: any, i: number) => (
                      <p key={i}>
                        {a.label}: {a.artifact.error}
                      </p>
                    ))}
                    <button className="link" onClick={() => openRun(v.run.id)}>
                      Inspect run {short(v.run.id)}
                    </button>
                  </details>
                )
              )
            })}
          </details>
        )}
      {current && data.length > 0 && (
        <ResultMatrix
          groups={groups}
          comparison={current.comparison}
          comparisonKey={comparisonKey}
          openRun={openRun}
        />
      )}{' '}
      {current &&
        Object.entries(groups).map(([groupId, values]) => (
          <ResultEvaluation
            key={groupId}
            groupId={groupId}
            values={values}
            comparison={current.comparison}
            onChange={(comparison) => void save({ ...current, comparison })}
            openRun={openRun}
          />
        ))}
      {draft && (
        <ResultCollectionDialog
          draft={draft}
          setDraft={setDraft}
          runs={runs}
          busy={busy}
          error={error}
          save={save}
          removeCollection={removeCollection}
        />
      )}
    </>
  )
}
