import { useEffect, useState } from 'react'
import Modal from './components/Modal'
import { api, short } from './api'
type Row = Record<string, any>
const labels: Record<string, string> = {
  generation: 'generations',
  scoring: 'scores',
  diagnostic: 'heuristic assessments',
  inspector: 'assessments',
  metrics: 'metric results',
  investigation: 'investigations',
  report: 'reports',
}
function dependencies(row: Row) {
  return [
    `${row.counts.runs} runs (${row.counts.active_runs} active)`,
    `${row.counts.items} response records`,
    ...Object.entries(row.job_types).map(([kind, n]) => `${n} ${labels[kind] || kind}`),
    `${row.counts.attempts} execution attempts`,
    `${row.counts.proposals} proposals`,
    `${row.counts.experiments} experiment records`,
    `${row.counts.result_memberships || 0} Result collections affected`,
    `${row.counts.shared_jobs} shared cached jobs retained`,
  ].join(' · ')
}
export default function DeleteRevisionsDialog({
  evaluationId,
  current,
  close,
  onDeleted,
}: {
  evaluationId: string
  current: string
  close: () => void
  onDeleted: (r: Row) => Promise<void>
}) {
  const [selected, setSelected] = useState([current]),
    [preview, setPreview] = useState<Row | null>(null),
    [rows, setRows] = useState<Row[]>([]),
    [error, setError] = useState(''),
    [busy, setBusy] = useState(false),
    [ack, setAck] = useState(false),
    [reload, setReload] = useState(0)
  useEffect(() => {
    let active = true
    setPreview(null)
    setAck(false)
    if (selected.length)
      void api(`/evaluations/${evaluationId}/deletion-preview`, { revision_ids: selected })
        .then((r) => {
          if (active) {
            setPreview(r)
            setRows(r.revisions)
          }
        })
        .catch((e) => {
          if (active) setError(String(e))
        })
    return () => {
      active = false
    }
  }, [evaluationId, selected, reload])
  async function remove() {
    if (!preview) return
    setBusy(true)
    setError('')
    try {
      const result = await api(`/evaluations/${evaluationId}/delete-revisions`, {
        revision_ids: selected,
        token: preview.token,
      })
      await onDeleted(result)
    } catch (e) {
      setError(String(e))
      setReload((n) => n + 1)
    } finally {
      setBusy(false)
    }
  }
  const hasData =
    preview && (preview.counts.runs || preview.counts.jobs || preview.counts.experiments)
  return (
    <Modal
      title="Delete revisions"
      close={() => {
        if (!busy) close()
      }}
      wide
    >
      <p>
        Select revisions to permanently delete. Each row shows its dependent data; shared cached
        jobs used by surviving runs are retained.
      </p>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      {!rows.length && !error && <p>Loading revisions and dependencies…</p>}
      {rows.map((row) => (
        <section className="panel" key={row.id}>
          <label className="check">
            <input
              type="checkbox"
              disabled={busy}
              checked={selected.includes(row.id)}
              onChange={(e) =>
                setSelected((ids) =>
                  e.target.checked ? [...ids, row.id] : ids.filter((id) => id !== row.id),
                )
              }
            />
            <strong>{row.name || short(row.id)}</strong> · {short(row.id)}
            {row.id === current ? ' · current revision' : ''}
          </label>
          <small>{new Date(row.created * 1000).toLocaleString()}</small>
          <p>{dependencies(row)}</p>
        </section>
      ))}
      {preview && (
        <>
          <h3>
            Selected: {selected.length} revision{selected.length === 1 ? '' : 's'}
          </h3>
          <p>{dependencies(preview)}</p>
          {hasData ? (
            <p role="alert" className="error">
              Dependent data listed above will also be permanently deleted. Active runs will stop;
              requests already sent cannot be undone.
            </p>
          ) : (
            <p>No run data depends on this selection.</p>
          )}
          {selected.length === rows.length && (
            <p>Deleting every revision also removes this evaluation from the library.</p>
          )}
          <p>
            Other revisions and runs remain. Links to deleted parents are cleared. Affected
            experiment records are removed, but their other runs remain.
          </p>
          <label className="check">
            <input
              type="checkbox"
              checked={ack}
              disabled={busy}
              onChange={(e) => setAck(e.target.checked)}
            />{' '}
            I understand that this deletion cannot be undone.
          </label>
        </>
      )}
      <div className="actions">
        <button disabled={busy} onClick={close}>
          Cancel
        </button>
        <button
          className="danger"
          disabled={busy || !preview || !ack || !selected.length}
          onClick={() => void remove()}
        >
          {busy ? 'Deleting…' : 'Delete selected revisions'}
        </button>
      </div>
    </Modal>
  )
}
