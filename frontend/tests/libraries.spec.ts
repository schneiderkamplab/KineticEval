import {test,expect} from '@playwright/test'

test('configure models and launch three evaluations into a Result without JSON',async({page,request})=>{
 const definition=await (await request.get('/api/demo')).json()
 for(const name of ['Matrix alpha','Matrix beta','Matrix gamma'])await request.post('/api/evaluations',{data:{...definition,name}})
 await page.goto('/')
 await page.getByRole('button',{name:'Models',exact:true}).click()
 for(const name of ['Variant A','Variant B','Variant C']){
  await page.getByRole('button',{name:'Add model',exact:true}).click()
  const dialog=page.getByRole('dialog',{name:'Add model',exact:true})
  await dialog.getByLabel('Configuration name').fill(name)
  await dialog.getByLabel('Model source').selectOption('fake')
  await dialog.getByLabel('Model identifier',{exact:true}).fill('same-physical-fixture')
  await dialog.getByRole('button',{name:'Save model',exact:true}).click()
  await expect(dialog).toHaveCount(0)
  await page.getByRole('checkbox',{name,exact:true}).check()
 }
 await page.getByRole('button',{name:'Create runs with selected models'}).click()
 const launch=page.getByRole('dialog',{name:'Create runs',exact:true})
 await launch.getByLabel('Result name').fill('Three by three')
 for(const name of ['Matrix alpha','Matrix beta','Matrix gamma'])await launch.getByRole('checkbox',{name,exact:true}).check()
 await expect(launch.getByText('3 runs × 3 models.',{exact:false})).toBeVisible()
 await expect(launch.locator('textarea[aria-label="Advanced JSON"]:visible')).toHaveCount(0)
 await launch.getByRole('button',{name:'Preview workload',exact:true}).click()
 await expect(launch.getByLabel('Workload preview')).toBeVisible()
 await expect(launch.getByLabel('Workload preview').getByText(/unique generation jobs: 3 new/)).toBeVisible()
 await launch.getByRole('button',{name:'Create Result and runs'}).click()
 const matrix=page.getByRole('table',{name:'Model by evaluation comparison'})
 await expect(matrix).toBeVisible()
 await expect(matrix.getByRole('columnheader')).toHaveCount(4)
 await expect(matrix.getByRole('rowheader')).toHaveCount(3)
 await expect(matrix.getByText('0.6667',{exact:true})).toHaveCount(9,{timeout:15000})
 const collections=await (await request.get('/api/results')).json()
 const collection=collections.find((r:any)=>r.name==='Three by three')
 expect(collection.run_ids).toHaveLength(3)
 for(const id of collection.run_ids){const run=await (await request.get('/api/runs/'+id)).json();expect(run.run.config.models).toHaveLength(3);expect(new Set(run.items.map((i:any)=>i.generation.id)).size).toBe(3)}
 await page.screenshot({path:'test-results/results-matrix.png',fullPage:true})
 await page.getByRole('button',{name:'Edit collection',exact:true}).click()
 const edit=page.getByRole('dialog',{name:'Edit Result collection'})
 await edit.getByRole('checkbox').first().uncheck()
 await edit.getByRole('button',{name:'Save collection'}).click()
 expect((await request.get('/api/runs/'+collection.run_ids[0])).status()).toBe(200)
})

test('upload JSONL and inspect the mapped preview without opening raw JSON',async({page})=>{
 await page.goto('/')
 await page.getByRole('button',{name:'Import definition',exact:false}).click()
 const dialog=page.getByRole('dialog',{name:'Import evaluation'})
 await dialog.getByLabel('Evaluation name').fill('Uploaded fixture')
 await dialog.getByLabel('Upload JSONL file').setInputFiles({name:'examples.jsonl',mimeType:'application/json',buffer:Buffer.from('{"id":"uploaded","input":"What is two plus two?","target":"4"}\n')})
 await expect(dialog.getByRole('cell',{name:'What is two plus two?',exact:true})).toBeVisible()
 await expect(dialog.getByLabel('JSONL data')).toBeHidden()
 await dialog.getByRole('button',{name:'Import',exact:true}).click()
 await expect(page.getByRole('heading',{name:'Uploaded fixture'})).toBeVisible()
})

test('model dialog supports keyboard dismissal and restores focus',async({page})=>{
 await page.goto('/')
 await page.getByRole('button',{name:'Models',exact:true}).click()
 const add=page.getByRole('button',{name:'Add model',exact:true})
 await add.click()
 const dialog=page.getByRole('dialog',{name:'Add model',exact:true})
 await expect(dialog).toBeFocused()
 await dialog.press('Tab')
 await expect(dialog.getByRole('button',{name:'Close',exact:true})).toBeFocused()
 await dialog.press('Escape')
 await expect(dialog).toHaveCount(0)
 await expect(add).toBeFocused()
})


test('select evaluation cards and open a launch with all selected revisions',async({page,request})=>{
 const definition=await (await request.get('/api/demo')).json()
 for(const name of ['Selected alpha','Selected beta'])await request.post('/api/evaluations',{data:{...definition,name}})
 await page.goto('/')
 await page.getByRole('checkbox',{name:'Select evaluation Selected alpha',exact:true}).check()
 await page.getByRole('checkbox',{name:'Select evaluation Selected beta',exact:true}).check()
 await page.getByRole('button',{name:'Create runs from selected evaluations (2)',exact:true}).click()
 const dialog=page.getByRole('dialog',{name:'Create runs',exact:true})
 await expect(dialog.getByRole('checkbox',{name:'Selected alpha',exact:true})).toBeChecked()
 await expect(dialog.getByRole('checkbox',{name:'Selected beta',exact:true})).toBeChecked()
 await expect(dialog.getByRole('button',{name:'Preview workload',exact:true})).toBeDisabled()
})

test('Result matrix refreshes active execution and completed scores without navigation',async({page})=>{
 let finished=false
 const model={name:'Live model',parameters:{max_tokens:64}}
 await page.route('**/api/results',route=>route.fulfill({json:[{id:'live-result',name:'Live Result',description:'',created:1,run_ids:['live-run'],comparison:{}}]}))
 await page.route('**/api/runs/live-run?summary=true',route=>route.fulfill({json:{
  run:{id:'live-run',evaluation_id:'live-evaluation',revision_id:'revision',status:finished?'completed':'active',manifest:{ids:['one','two']},config:{models:[model],repeats:1}},
  definition:{name:'Live evaluation',scorer:{name:'exact'},metrics:[{name:'accuracy'}]},
  summaries:[{model:'Live model',total:2,scored:finished?2:0,scored_correct:finished?1:0,reused:0,metrics:finished?{accuracy:.5}:{},metric_errors:[],metric_status:finished?'succeeded':'waiting',inspector_assessed:0,execution:{generation:finished?{succeeded:2}:{running:1,pending:1}}}],report:null,investigations:[]
 }}))
 await page.route('**/api/runs/live-run/progress',route=>route.fulfill({status:503,json:{detail:'fixture'}}))
 await page.goto('/')
 await page.getByRole('button',{name:'Results',exact:true}).click()
 await page.getByRole('combobox',{name:'Result',exact:true}).selectOption('live-result')
 const matrix=page.getByRole('table',{name:'Model by evaluation comparison'})
 await expect(matrix.getByText('Running · generating 1',{exact:true})).toBeVisible()
 finished=true
 await expect(matrix.getByText('0.5000',{exact:true})).toBeVisible({timeout:12000})
 await expect(matrix.getByText('accuracy · 2/2 scored')).toBeVisible()
})

test('save a custom base-model chat template without JSON',async({page,request})=>{
 await page.goto('/')
 await page.getByRole('button',{name:'Models',exact:true}).click()
 await page.getByRole('button',{name:'Add model',exact:true}).click()
 const dialog=page.getByRole('dialog',{name:'Add model',exact:true})
 await dialog.getByLabel('Configuration name').fill('Base model with plain messages')
 await dialog.getByLabel('Model identifier',{exact:true}).fill('sapientinc/HRM-Text-1B')
 await dialog.getByLabel('Use a custom chat template').check()
 await expect(dialog.getByLabel('Custom chat template (Jinja)')).toContainText('message')
 const template=await dialog.getByLabel('Custom chat template (Jinja)').inputValue()
 await dialog.getByRole('button',{name:'Save model',exact:true}).click()
 await expect(dialog).toHaveCount(0)
 const rows=await (await request.get('/api/models')).json()
 expect(rows.find((r:any)=>r.name==='Base model with plain messages').spec.chat_template).toBe(template)
 await page.locator('article').filter({hasText:'Base model with plain messages'}).getByRole('button',{name:'Edit defaults'}).click()
 await expect(page.getByLabel('Custom chat template (Jinja)')).toHaveValue(template)
})
