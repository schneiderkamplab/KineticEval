import Modal from './components/Modal'
import { short } from './api'
type Row = Record<string, any>
export default function ResultCollectionDialog({
  draft,
  setDraft,
  runs,
  busy,
  error,
  save,
  removeCollection,
}: {
  draft: Row
  setDraft: (value: Row | null) => void
  runs: Row[]
  busy: boolean
  error: string
  save: (value: Row) => Promise<void>
  removeCollection: () => Promise<void>
}) {
  return (
    <Modal
      title="Edit Result collection"
      close={() => {
        if (!busy) setDraft(null)
      }}
    >
      {error && <p role="alert">{error}</p>}
      <label>
        Result name
        <input value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
      </label>
      <label>
        Description
        <textarea
          value={draft.description}
          onChange={(e) => setDraft({ ...draft, description: e.target.value })}
        />
      </label>
      {runs.map((r) => (
        <label className="check" key={r.id}>
          <input
            type="checkbox"
            checked={draft.run_ids.includes(r.id)}
            onChange={(e) =>
              setDraft({
                ...draft,
                run_ids: e.target.checked
                  ? [...draft.run_ids, r.id]
                  : draft.run_ids.filter((id: string) => id !== r.id),
              })
            }
          />
          {short(r.id)} · {r.config.models.map((m: Row) => m.name).join(', ')} · {r.status}
        </label>
      ))}
      <p>Removing runs here only changes collection membership.</p>
      <button disabled={busy || !draft.name.trim()} onClick={() => void save(draft)}>
        Save collection
      </button>
      {draft.id && (
        <button className="danger" disabled={busy} onClick={() => void removeCollection()}>
          Delete collection only
        </button>
      )}
    </Modal>
  )
}
