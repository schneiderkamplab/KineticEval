import { describe, it, expect } from 'vitest'
import { recentModels, validateModels } from './runConfig'
describe('evaluation model configuration', () => {
  it('excludes inspector providers and retains independent pinned local configurations', () => {
    const model = {
      name: 'local-model',
      backend: 'hf',
      local_path: '/weights/export',
      artifact_hash: 'abc',
      revision: 'pinned',
      parameters: { max_tokens: 128 },
      chat_template_kwargs: { enable_thinking: false },
    }
    const history = [
      { config: { models: [model, { name: 'inspector', backend: 'ucloud', parameters: {} }] } },
      { config: { models: [{ ...model, revision: 'old' }] } },
    ]
    const models = recentModels(history)
    expect(models).toEqual([model])
    models[0].parameters.max_tokens = 256
    expect(model.parameters.max_tokens).toBe(128)
  })
  it('validates model selection and context capacity', () => {
    expect(validateModels([])).toMatch(/Select/)
    expect(validateModels([{ name: 'inspector', backend: 'ucloud', parameters: {} }])).toMatch(
      /reserved/,
    )
    expect(
      validateModels([
        { name: 'hf-model', backend: 'hf', context_limit: 128, parameters: { max_tokens: 128 } },
      ]),
    ).toMatch(/prompt/)
    expect(
      validateModels([{ name: 'hf-model', backend: 'hf', parameters: { max_tokens: 64 } }]),
    ).toBeNull()
  })
})
