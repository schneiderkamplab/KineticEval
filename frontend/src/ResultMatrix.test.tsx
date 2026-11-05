import {render,screen} from '@testing-library/react'
import {describe,it,expect,vi} from 'vitest'
import {ResultMatrix,ExecutionStatus} from './ResultMatrix'
import {comparisonKey,ranking} from './Results'
function run(id:string,revision='rev',scores=[1,.5,.5]){return {run:{id,revision_id:revision,manifest:{ids:['x']},config:{repeats:1,models:scores.map((_,i)=>({name:'model'+i,parameters:{}}))}},definition:{name:'Eval',scorer:{name:'exact',config:{}},metrics:[{name:'accuracy'}]},summaries:scores.map((score,i)=>({model:'model'+i,scored:1,total:1,metrics:{accuracy:score}}))}}
describe('Results ranking',()=>{
 it('uses distinct ranks and ties in the matrix',()=>{render(<ResultMatrix groups={{eval:[run('one')]}} comparison={{}} comparisonKey={comparisonKey} openRun={vi.fn()}/>);const table=screen.getByRole('table');expect(table.querySelectorAll('strong')).toHaveLength(1);expect(table.querySelectorAll('u')).toHaveLength(2)})
 it('sorts lower-is-better and ignores object key insertion order',()=>{expect(ranking([2,1,1,3],'lower')).toEqual([1,2,3]);const a=run('one'),b=run('two');b.definition.scorer={config:{},name:'exact'};expect(comparisonKey(a)).toBe(comparisonKey(b))})
 it('keeps repeated runs visible and warns about mismatches',()=>{render(<ResultMatrix groups={{eval:[run('one'),run('two','other')]}} comparison={{allow_mismatch:true}} comparisonKey={comparisonKey} openRun={vi.fn()}/>);expect(screen.getByRole('alert')).toHaveTextContent('Mismatch override enabled');expect(screen.getAllByRole('button',{name:'Run two'})).toHaveLength(3)})
})

it('does not rank unknown metric directions until explicitly configured',()=>{
 const value=run('unknown');value.summaries=value.summaries.map((s:any)=>({...s,metrics:{custom_score:5}}))
 const {rerender}=render(<ResultMatrix groups={{eval:[value]}} comparison={{}} comparisonKey={comparisonKey} openRun={vi.fn()}/>)
 expect(screen.getByRole('table').querySelectorAll('strong')).toHaveLength(0)
 rerender(<ResultMatrix groups={{eval:[value]}} comparison={{directions:{eval:'lower'}}} comparisonKey={comparisonKey} openRun={vi.fn()}/>)
 expect(screen.getByRole('table').querySelectorAll('strong')).toHaveLength(3)
})


it('distinguishes active work from queued and waiting models',()=>{
 const {rerender}=render(<ExecutionStatus summary={{execution:{generation:{running:3,pending:2}}}} status="active"/>)
 expect(screen.getByText('Running · generating 3')).toBeVisible()
 rerender(<ExecutionStatus summary={{execution:{generation:{pending:2}}}} status="active"/>)
 expect(screen.getByText('Queued · generating 2')).toBeVisible()
 rerender(<ExecutionStatus summary={{execution:{scoring:{waiting:2}}}} status="active"/>)
 expect(screen.getByText('Waiting for earlier stages')).toBeVisible()
})
