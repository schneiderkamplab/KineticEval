import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'

type Row = Record<string, any>

/** Poll the selected collection without mixing network errors with action errors. */
export function useResultData(selected: string) {
  const [rows, setRows] = useState<Row[]>([])
  const [snapshot, setSnapshot] = useState<{ id: string; data: Row[] }>({ id: '', data: [] })
  const [error, setError] = useState('')
  const [version, setVersion] = useState(0)
  const refresh = useCallback(() => setVersion((value) => value + 1), [])

  useEffect(() => {
    let active = true
    let timer: ReturnType<typeof setTimeout>
    async function load() {
      try {
        const collections = await api('/results')
        const collection = collections.find((row: Row) => row.id === selected)
        const data = collection
          ? await Promise.all(
              collection.run_ids.map((id: string) => api(`/runs/${id}?summary=true`)),
            )
          : []
        if (active) {
          setRows(collections)
          setSnapshot({ id: selected, data })
          setError('')
        }
      } catch (error) {
        if (active) setError(String(error))
      } finally {
        if (active) timer = setTimeout(load, 5000)
      }
    }
    void load()
    return () => {
      active = false
      clearTimeout(timer)
    }
  }, [selected, version])

  return { rows, data: snapshot.id === selected ? snapshot.data : [], error, refresh }
}
