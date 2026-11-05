import WorkloadPreview from './WorkloadPreview'
import { useEffect, useState, useRef } from 'react'
import { api, short } from './api'
import Modal from './components/Modal'
import { DefinitionFields, withDefaults } from './Evaluation'
import { ModelFields } from './Models'
type Row = Record<string, any>
export default function LaunchDialog({
  evaluations,
  initialRevisions = [],
  initialModels = [],
  close,
  onStarted,
}: {
  evaluations: Row[]
  initialRevisions?: string[]
  initialModels?: string[]
  close: () => void
  onStarted: (id: string) => Promise<void>
}) {
  const [models, setModels] = useState<Row[]>([]),
    [chosen, setChosen] = useState<string[]>(initialModels),
    [revisions, setRevisions] = useState<string[]>(initialRevisions),
    [definitions, setDefinitions] = useState<Record<string, Row>>({}),
    [name, setName] = useState('Evaluation results'),
    [fresh, setFresh] = useState(false),
    [repeats, setRepeats] = useState(1),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(''),
    [key] = useState(() => crypto.randomUUID()),
    [adding, setAdding] = useState(false)
  const [estimate, setEstimate] = useState<Row | null>(null),
    [previewing, setPreviewing] = useState(false)
  const [newModel, setNewModel] = useState<Row>({
    name: 'New model',
    spec: {
      name: '',
      backend: 'hf',
      parameters: { max_tokens: 64, temperature: 0 },
      chat_template_kwargs: { enable_thinking: false },
    },
  })
  useEffect(() => {
    void api('/models')
      .then(setModels)
      .catch((e) => setError(String(e)))
  }, [])
  useEffect(() => {
    let active = true
    void Promise.all(
      revisions
        .filter((id) => !definitions[id])
        .map(async (id) => [id, withDefaults(await api('/revisions/' + id))] as const),
    )
      .then((pairs) => {
        if (active && pairs.length) setDefinitions((d) => ({ ...d, ...Object.fromEntries(pairs) }))
      })
      .catch((e) => setError(String(e)))
    return () => {
      active = false
    }
  }, [revisions, definitions])
  const ready =
    !!revisions.length &&
    !!chosen.length &&
    chosen.every((id) => models.some((m) => m.id === id)) &&
    revisions.every((id) => definitions[id])
  const requests = ready
    ? revisions.map((id) => ({
        revision_id: id,
        models: models
          .filter((m) => chosen.includes(m.id))
          .map((m) => ({
            ...m.spec,
            model_identifier: m.spec.name,
            name: m.name,
            configuration_id: m.id,
          })),
        repeats,
        fresh,
        analysis_overrides: Object.fromEntries(
          [
            'generation_defaults',
            'scorer',
            'metrics',
            'heuristics',
            'inspector',
            'expected_format',
          ].map((k) => [k, definitions[id][k]]),
        ),
      }))
    : []
  const requestKey = JSON.stringify(requests),
    requestRef = useRef(requestKey)
  requestRef.current = requestKey
  useEffect(() => {
    setEstimate(null)
  }, [requestKey])
  async function preview() {
    setPreviewing(true)
    setError('')
    const key = requestKey
    try {
      const result = await api('/launches/preview', { runs: requests })
      if (requestRef.current === key) setEstimate(result)
    } catch (e) {
      if (requestRef.current === key) setError(String(e))
    } finally {
      setPreviewing(false)
    }
  }
  async function submit() {
    setBusy(true)
    setError('')
    try {
      const result = await api('/launches', {
        name,
        launch_key: key,
        runs: estimate?.resolved_runs || requests,
      })
      await onStarted(result.id)
    } catch (e) {
      setError(String(e))
      setBusy(false)
    }
  }

  return (
    <Modal
      title="Create runs"
      close={() => {
        if (!busy) close()
      }}
      wide
    >
      <p>One run per evaluation, with all selected models. Runs are grouped into one Result.</p>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      <fieldset disabled={busy}>
        <label>
          Result name
          <input value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <h3>Evaluations and revisions</h3>
        {evaluations.map((e) => {
          const selected = e.revisions.find((r: Row) => revisions.includes(r.id))
          return (
            <section className="panel" key={e.id}>
              <label className="check">
                <input
                  type="checkbox"
                  checked={!!selected}
                  onChange={(event) =>
                    setRevisions((ids) =>
                      event.target.checked
                        ? [...ids, e.revisions[0].id]
                        : ids.filter((id) => !e.revisions.some((r: Row) => r.id === id)),
                    )
                  }
                />
                {e.name}
              </label>
              {selected && (
                <>
                  <label>
                    Revision for {e.name}
                    <select
                      value={selected.id}
                      onChange={(event) =>
                        setRevisions((ids) =>
                          ids.map((id) => (id === selected.id ? event.target.value : id)),
                        )
                      }
                    >
                      {e.revisions.map((r: Row) => (
                        <option key={r.id} value={r.id}>
                          {r.name || short(r.id)} · {new Date(r.created * 1000).toLocaleString()}
                        </option>
                      ))}
                    </select>
                  </label>
                  {definitions[selected.id] ? (
                    <>
                      <p>
                        {definitions[selected.id].examples.length} available examples ·{' '}
                        {definitions[selected.id].sampler.limit || 'all'} selected by sampler
                      </p>
                      <details>
                        <summary>Scoring and diagnostic settings</summary>
                        <DefinitionFields
                          section="scoring"
                          value={definitions[selected.id]}
                          onChange={(d) => setDefinitions((old) => ({ ...old, [selected.id]: d }))}
                        />
                      </details>
                      <details>
                        <summary>LLM inspector settings</summary>
                        <DefinitionFields
                          section="inspector"
                          value={definitions[selected.id]}
                          onChange={(d) => setDefinitions((old) => ({ ...old, [selected.id]: d }))}
                        />
                      </details>
                    </>
                  ) : (
                    <p>Loading settings…</p>
                  )}
                </>
              )}
            </section>
          )
        })}
        <h3>Model configurations</h3>
        {models.map((m) => (
          <section className="panel" key={m.id}>
            <label className="check">
              <input
                type="checkbox"
                checked={chosen.includes(m.id)}
                onChange={(e) =>
                  setChosen((ids) =>
                    e.target.checked ? [...ids, m.id] : ids.filter((id) => id !== m.id),
                  )
                }
              />
              {m.name} · {m.spec.name}
            </label>
            {chosen.includes(m.id) && (
              <details>
                <summary>Adjust settings for these runs</summary>
                <ModelFields
                  value={m.spec}
                  onChange={(spec) =>
                    setModels((rows) =>
                      rows.map((row) => (row.id === m.id ? { ...row, spec } : row)),
                    )
                  }
                />
              </details>
            )}
          </section>
        ))}
        <button onClick={() => setAdding(!adding)}>Add model configuration</button>
        {adding && (
          <section className="panel">
            <label>
              Configuration name
              <input
                value={newModel.name}
                onChange={(e) => setNewModel({ ...newModel, name: e.target.value })}
              />
            </label>
            <ModelFields
              value={newModel.spec}
              onChange={(spec) => setNewModel({ ...newModel, spec })}
            />
            <button
              disabled={!newModel.name.trim() || !newModel.spec.name.trim()}
              onClick={() =>
                void api('/models', newModel)
                  .then((row) => {
                    setModels((old) => [...old, row])
                    setChosen((ids) => [...ids, row.id])
                    setAdding(false)
                  })
                  .catch((e) => setError(String(e)))
              }
            >
              Save and select model
            </button>
          </section>
        )}
        <label>
          Repeats per example
          <input
            type="number"
            min={1}
            max={100}
            value={repeats}
            onChange={(e) => setRepeats(Number(e.target.value))}
          />
        </label>
        <label className="check">
          <input type="checkbox" checked={fresh} onChange={(e) => setFresh(e.target.checked)} />{' '}
          Generate fresh responses instead of reusing matching generations
        </label>
        <p>
          {revisions.length} runs × {chosen.length} models. Editing settings here does not change
          saved defaults.
        </p>
      </fieldset>
      <button disabled={!ready || previewing || busy} onClick={() => void preview()}>
        {previewing ? 'Checking cache and model revisions…' : 'Preview workload'}
      </button>
      {estimate ? (
        <WorkloadPreview value={estimate} />
      ) : (
        <p className="field-help">
          Preview workload to see reusable generations, new work, and bounded LLM inspection
          estimates. Previewing queues no work.
        </p>
      )}
      <div className="actions">
        <button disabled={busy} onClick={close}>
          Cancel
        </button>
        <button
          className="primary"
          disabled={
            busy ||
            previewing ||
            !name.trim() ||
            !ready ||
            !Number.isInteger(repeats) ||
            repeats < 1 ||
            repeats > 100
          }
          onClick={() => void submit()}
        >
          {busy ? 'Creating…' : 'Create Result and runs'}
        </button>
      </div>
    </Modal>
  )
}
