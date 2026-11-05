import {render,screen,fireEvent,waitFor} from '@testing-library/react'
import {it,expect,vi} from 'vitest'
import RunDialog from './RunDialog'
import {withDefaults} from './Evaluation'
import {api} from './api'
vi.mock('./api',async importOriginal=>({...await importOriginal<typeof import('./api')>(),api:vi.fn()}))
it('queues a typed HF identifier as a local model without querying inspector discovery',async()=>{
 const definition=withDefaults({name:'QA',examples:[{id:'a',input:'Q',target:'A',subset:'test'}]})
 vi.mocked(api).mockResolvedValue({id:'new-run'})
 const started=vi.fn().mockResolvedValue(undefined)
 render(<RunDialog revision={{id:'revision',definition}} evaluation={{revisions:[{id:'revision'}]}} history={[]} close={()=>{}} onStarted={started}/>)
 fireEvent.change(screen.getByLabelText('Hugging Face model identifier'),{target:{value:'organization/model-name'}})
 fireEvent.change(screen.getByLabelText('Weight revision (optional)'),{target:{value:'pinned-sha'}})
 fireEvent.click(screen.getByText('Add to this run'))
 fireEvent.click(screen.getByRole('button',{name:'Start run'}))
 await waitFor(()=>expect(started).toHaveBeenCalledWith('new-run'))
 expect(api).toHaveBeenCalledTimes(1)
 expect(api).toHaveBeenCalledWith('/runs',expect.objectContaining({revision_id:'revision',fresh:false,models:[expect.objectContaining({name:'organization/model-name',backend:'hf',revision:'pinned-sha'})]}))
})
