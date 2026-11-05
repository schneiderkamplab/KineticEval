type Row = Record<string, any>
function cacheCounts(counts: Row) {
  return (
    ['reusable', 'new', 'queued', 'running', 'failed']
      .filter((key) => counts[key])
      .map(
        (key) =>
          `${counts[key]} ${key === 'reusable' ? 'reusable' : key === 'failed' ? 'previously failed' : key}`,
      )
      .join(' · ') || 'None'
  )
}
export default function WorkloadPreview({ value }: { value: Row }) {
  return (
    <section className="panel" aria-label="Workload preview">
      <h3>Estimated work</h3>
      <p>
        <strong>{value.response_slots.toLocaleString()} response slots</strong> · unique generation
        jobs: {cacheCounts(value.generation_jobs)}
      </p>
      <p>
        Shared generations are counted once in the total, even when several configurations or
        evaluations use them.
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th>Evaluation / model</th>
              <th>Generations</th>
              <th>Scoring</th>
              <th>Heuristics</th>
              <th>LLM inspections</th>
            </tr>
          </thead>
          <tbody>
            {value.groups.map((g: Row, i: number) => (
              <tr key={i}>
                <td>
                  {g.evaluation}
                  <small>
                    {g.model} · {g.responses} responses
                  </small>
                </td>
                <td>{cacheCounts(g.generation)}</td>
                <td>
                  {cacheCounts(g.scoring)}
                  {g.scoring.unknown > 0 && (
                    <small>{g.scoring.unknown} depend on new responses</small>
                  )}
                </td>
                <td>
                  {cacheCounts(g.heuristics)}
                  {g.heuristics.unknown > 0 && (
                    <small>{g.heuristics.unknown} depend on scoring</small>
                  )}
                </td>
                <td>
                  {g.inspection.exact ? (
                    cacheCounts(g.inspection.counts)
                  ) : (
                    <>
                      <strong>0–{g.inspection.maximum}</strong> new inspections
                      <small>
                        Upper bound; depends on outcomes, heuristics, audit selection, and cache
                        reuse.
                      </small>
                    </>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        Additionally: up to{' '}
        {value.groups.reduce((n: number, g: Row) => n + g.investigation_maximum, 0)} pattern
        investigations and {value.reports_maximum} reports. Existing matching results may be reused.
      </p>
      <p className="field-help">{value.note}</p>
    </section>
  )
}
