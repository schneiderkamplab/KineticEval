export async function api(path: string, body?: unknown): Promise<any> {
  const response = await fetch(
    '/api' + path,
    body === undefined
      ? {}
      : {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        },
  )
  const result = await response.json()
  if (!response.ok)
    throw new Error(
      typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail || result),
    )
  return result
}
export const pretty = (value: unknown) => JSON.stringify(value, null, 2)
export const short = (id: string) => id.slice(0, 8)
export const percent = (n: number | null | undefined) =>
  n == null ? '—' : `${Math.round(n * 100)}%`
