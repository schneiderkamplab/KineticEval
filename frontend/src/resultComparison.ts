import { canonical } from './canonical'
type Row = Record<string, any>
export function comparisonKey(value: Row) {
  return JSON.stringify(
    canonical([
      value.run.revision_id,
      [...value.run.manifest.ids].sort(),
      value.run.config.repeats,
      value.definition.scorer,
      value.definition.metrics,
    ]),
  )
}
export function ranking(values: number[], direction: string) {
  return [...new Set(values)].sort((a, b) => (direction === 'lower' ? a - b : b - a))
}
