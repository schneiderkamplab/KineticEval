import {test,expect} from '@playwright/test'
test('experiment preview requires explicit approval and edits invalidate the preview',async({page})=>{
 let prepared=0,approved=0
 const proposal={kind:'experiment',title:'Compare token budgets',rationale:'Judge recommendation',patch:{},origin_job:'judge-job'}
 const plan={model:'fixture',example_ids:['one','two'],arms:[{name:'cap128',parameters:{max_tokens:128}}]}
 const record:any={id:'experiment1',proposal_id:'proposal1',baseline_id:'baseline1',status:'prepared',plan,runs:{},preview:{proposal_snapshot:proposal,origin_job:'judge-job',baseline_model:{name:'fixture'},note:'Only approval queues runs.',arms:[{name:'cap128',examples:2,repeats:1,reusable:0,pending_or_new:2,output_token_ceiling:256}]}}
 await page.route('**/api/**',async route=>{
  const path=new URL(route.request().url()).pathname
  let data:any=[]
  if(path==='/api/experiment-proposals')data=[{id:'proposal1',status:'pending',proposal}]
  if(path==='/api/proposals/proposal1/experiment')data={proposal,status:'pending',baseline_id:'baseline1',plan,models:[{name:'fixture'}],warnings:[],scope_note:'Matched examples'}
  if(path==='/api/proposals/proposal1/prepare-experiment'){prepared++;data=record}
  if(path==='/api/experiments/experiment1/approve'){approved++;record.status='approved';record.approved_at=1700000000;data=record}
  if(path==='/api/experiments/experiment1')data=record
  await route.fulfill({json:data})
 })
 await page.goto('/')
 await page.getByRole('button',{name:'Experiments',exact:true}).click()
 await page.getByRole('button',{name:'Configure experiment proposal'}).click()
 await page.getByRole('button',{name:'Save approval preview'}).click()
 await expect(page.getByRole('button',{name:'Approve and queue follow-up runs'})).toBeVisible()
 expect(prepared).toBe(1);expect(approved).toBe(0)
 await page.getByLabel('Experiment arms').fill(JSON.stringify([{name:'cap256',parameters:{max_tokens:256}}]))
 await expect(page.getByRole('button',{name:'Approve and queue follow-up runs'})).toHaveCount(0)
 await page.getByRole('button',{name:'Save approval preview'}).click()
 await page.getByRole('button',{name:'Approve and queue follow-up runs'}).click()
 await expect(page.getByText(/Runs are queued for workers/)).toBeVisible()
 expect(approved).toBe(1)
 await page.screenshot({path:'test-results/experiment-approval.png',fullPage:true})
})
