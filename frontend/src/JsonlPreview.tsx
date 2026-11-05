export default function JsonlPreview({
  text,
  mapping,
}: {
  text: string
  mapping: Record<string, string>
}) {
  if (!text.trim())
    return <p>Upload a JSONL file to preview its questions and reference answers.</p>
  try {
    const lines = text.trim().split('\n')
    const rows = lines.slice(0, 5).map((line) => JSON.parse(line))
    return (
      <>
        <p>
          {lines.length} rows supplied · previewing {rows.length}
        </p>
        <table>
          <thead>
            <tr>
              <th>Identifier</th>
              <th>Question</th>
              <th>Reference</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                <td>{String(row[mapping.id || 'id'] ?? i + 1)}</td>
                <td>{String(row[mapping.input || 'input'] ?? 'Missing question column')}</td>
                <td>{String(row[mapping.target || 'target'] ?? 'Missing reference column')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </>
    )
  } catch {
    return (
      <p role="alert">
        The preview could not read this JSONL. Check that each line is one valid example.
      </p>
    )
  }
}
