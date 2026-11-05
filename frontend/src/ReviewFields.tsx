import { useState } from 'react'
import { ReadableValue } from './components/StructuredValue'
type Row = Record<string, any>
export function ProposalChanges({ definition, patch }: { definition: Row; patch: Row }) {
  return (
    <section>
      <h3>Proposed changes</h3>
      {Object.entries(patch).map(([key, value]) => (
        <section className="panel" key={key}>
          <h4>{key.replaceAll('_', ' ')}</h4>
          <div className="two-columns">
            <div>
              <h5>Before</h5>
              <ReadableValue value={definition[key]} />
            </div>
            <div>
              <h5>Proposed</h5>
              <ReadableValue value={value} />
            </div>
          </div>
        </section>
      ))}
    </section>
  )
}
export function RegressionCases({
  value,
  onChange,
}: {
  value: Row[]
  onChange: (v: Row[]) => void
}) {
  const update = (i: number, patch: Row) =>
    onChange(value.map((row, j) => (i === j ? { ...row, ...patch } : row)))
  return (
    <section>
      <h3>Regression cases</h3>
      <p>Include both positive examples and counterexamples.</p>
      {value.map((row, i) => (
        <fieldset key={i}>
          <legend>Case {i + 1}</legend>
          <label>
            Response
            <textarea
              value={row.response}
              onChange={(e) => update(i, { response: e.target.value })}
            />
          </label>
          <label>
            Reference answer
            <input value={row.target} onChange={(e) => update(i, { target: e.target.value })} />
          </label>
          <label>
            Prompt (optional)
            <textarea
              value={row.prompt || ''}
              onChange={(e) => update(i, { prompt: e.target.value })}
            />
          </label>
          <label>
            Expected semantic correctness
            <select
              value={row.expected_semantic === null ? 'unknown' : String(row.expected_semantic)}
              onChange={(e) =>
                update(i, {
                  expected_semantic:
                    e.target.value === 'unknown' ? null : e.target.value === 'true',
                })
              }
            >
              <option value="true">Correct</option>
              <option value="false">Incorrect</option>
              <option value="unknown">Uncertain</option>
            </select>
          </label>
          <button onClick={() => onChange(value.filter((_, j) => i !== j))}>
            Remove case {i + 1}
          </button>
        </fieldset>
      ))}
      <button
        onClick={() => onChange([...value, { response: '', target: '', expected_semantic: null }])}
      >
        Add regression case
      </button>
    </section>
  )
}
export function ExperimentArms({
  value,
  onChange,
}: {
  value: Row[]
  onChange: (v: Row[]) => void
}) {
  const update = (i: number, patch: Row) =>
    onChange(value.map((row, j) => (i === j ? { ...row, ...patch } : row)))
  return (
    <section>
      <h3>Experiment arms</h3>
      {value.map((arm, i) => (
        <fieldset key={i}>
          <legend>Arm {i + 1}</legend>
          <label>
            Arm name
            <input value={arm.name} onChange={(e) => update(i, { name: e.target.value })} />
          </label>
          <div className="form-columns">
            {[
              ['max_tokens', 'Maximum output tokens'],
              ['temperature', 'Temperature'],
              ['top_p', 'Top p'],
              ['seed', 'Seed'],
            ].map(([key, label]) => (
              <label key={key}>
                {label}
                <input
                  type="number"
                  step={['max_tokens', 'seed'].includes(key) ? 1 : 0.01}
                  placeholder="Inherit baseline"
                  value={arm.parameters?.[key] ?? ''}
                  onChange={(e) => {
                    const parameters = { ...arm.parameters }
                    if (e.target.value === '') delete parameters[key]
                    else parameters[key] = Number(e.target.value)
                    update(i, { parameters })
                  }}
                />
              </label>
            ))}
          </div>
          <label>
            Thinking mode
            <select
              value={
                arm.chat_template_kwargs?.enable_thinking === undefined
                  ? 'inherit'
                  : String(arm.chat_template_kwargs.enable_thinking)
              }
              onChange={(e) => {
                const options = { ...arm.chat_template_kwargs }
                if (e.target.value === 'inherit') delete options.enable_thinking
                else options.enable_thinking = e.target.value === 'true'
                update(i, { chat_template_kwargs: options })
              }}
            >
              <option value="inherit">Inherit baseline</option>
              <option value="true">Request thinking</option>
              <option value="false">Disable thinking</option>
            </select>
          </label>
          <label>
            Prompt template override (optional)
            <textarea
              value={arm.prompt_template ?? ''}
              placeholder="Inherit baseline"
              onChange={(e) => update(i, { prompt_template: e.target.value || null })}
            />
          </label>
          <button onClick={() => onChange(value.filter((_, j) => i !== j))}>
            Remove arm {i + 1}
          </button>
        </fieldset>
      ))}
      <button
        onClick={() =>
          onChange([
            ...value,
            { name: `arm-${value.length + 1}`, parameters: {}, chat_template_kwargs: {} },
          ])
        }
      >
        Add experiment arm
      </button>
    </section>
  )
}
export function CopyText({ text }: { text: string }) {
  const [status, setStatus] = useState('')
  return (
    <div>
      <pre>{text}</pre>
      <button
        onClick={() =>
          void navigator.clipboard
            .writeText(text)
            .then(() => setStatus('Copied'))
            .catch(() => setStatus('Copy unavailable; select the text to copy.'))
        }
      >
        Copy text
      </button>
      {status && <small role="status">{status}</small>}
    </div>
  )
}
