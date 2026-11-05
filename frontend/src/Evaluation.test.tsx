import {render,screen,fireEvent} from '@testing-library/react'
import {describe,it,expect,vi} from 'vitest'
import {EvaluationOverview,DefinitionFields,withDefaults} from './Evaluation'
const definition=()=>withDefaults({name:'Example',examples:[{id:'a',input:'Q',target:'A',choices:['a','b'],subset:'one'},{id:'b',input:'Q2',target:'A',choices:['a','b'],subset:'two'}],overlays:{a:{target:'B'}},sampler:{limit:1,subsets:[],seed:42},scorer:{name:'inspect',version:'1',config:{scorer_spec:{name:'inspect_ai/choice'},pinned:'keep me'}},metrics:[{name:'inspect',version:'1',config:{metric_names:['inspect_ai/accuracy','inspect_ai/stderr']}}]})
describe('evaluation presentation',()=>{
 it('shows examples, corrections, format and actual imported scorer without JSON',()=>{
  const {container}=render(<EvaluationOverview definition={definition()}/>)
  expect(screen.getByText('Corrected examples')).toBeInTheDocument()
  expect(screen.getByText('50.0% corrected · original data preserved')).toBeInTheDocument()
  expect(screen.getByText('Inspect · Multiple choice')).toBeInTheDocument()
  expect(screen.getByText('Multiple-choice QA')).toBeInTheDocument()
  expect(container.querySelector('pre')).toBeNull()
 })
 it('preserves imported component configuration when editing sampling',()=>{
  const d=definition(),changed=vi.fn()
  render(<DefinitionFields value={d} onChange={changed}/>)
  fireEvent.change(screen.getByLabelText('Maximum examples'),{target:{value:'2'}})
  const value=changed.mock.calls[0][0]
  expect(value.sampler.limit).toBe(2)
  expect(value.scorer).toEqual(d.scorer)
  expect(value.metrics).toEqual(d.metrics)
  expect(value.source).toEqual(d.source)
 })
})
