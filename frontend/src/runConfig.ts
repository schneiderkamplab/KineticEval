export type Model = Record<string, any> & {
  name: string
  backend: string
  parameters: Record<string, any>
}
export const demoModels: Model[] = [
  { name: 'fixture-literal', backend: 'fake', parameters: {} },
  { name: 'fixture-verbose', backend: 'fake', parameters: { style: 'sentence' } },
]
export function recentModels(runs: Record<string, any>[]): Model[] {
  const seen = new Set<string>()
  const models: Model[] = []
  for (const run of runs)
    for (const model of run.config.models) {
      if (model.backend === 'ucloud' || seen.has(model.name)) continue
      seen.add(model.name)
      models.push(structuredClone(model))
    }
  return models
}
export function validateModels(models: Model[]): string | null {
  if (!models.length) return 'Select at least one model.'
  if (new Set(models.map((m) => m.name)).size !== models.length)
    return 'Each model must have a unique name.'
  for (const model of models) {
    if (!model.name.trim()) return 'Enter a model identifier.'
    if (model.backend === 'ucloud')
      return 'UCloud is reserved for inspector configuration in this workspace.'
    const p = model.parameters || {}
    if (p.max_tokens !== undefined && (!Number.isInteger(p.max_tokens) || p.max_tokens < 1))
      return `${model.name}: maximum output tokens must be a positive integer.`
    if (
      p.temperature !== undefined &&
      (!Number.isFinite(p.temperature) || p.temperature < 0 || p.temperature > 2)
    )
      return `${model.name}: temperature must be between 0 and 2.`
    if (model.backend === 'hf' && p.max_tokens >= (model.context_limit || 4096))
      return `${model.name}: output tokens must leave room for the prompt in the context window.`
  }
  return null
}
