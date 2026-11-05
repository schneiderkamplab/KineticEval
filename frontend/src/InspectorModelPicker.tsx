import {useEffect,useState} from 'react'
import {api} from './api'

export default function InspectorModelPicker({value,onChange}:{value:string;onChange:(value:string)=>void}){
 const [models,setModels]=useState<string[]>([]),[loading,setLoading]=useState(true),[failed,setFailed]=useState(false),[refresh,setRefresh]=useState(0)
 useEffect(()=>{
  let active=true
  setLoading(true);setFailed(false)
  api('/providers/ucloud/models').then(result=>{if(active)setModels([...new Set(result as string[])].sort())})
   .catch(()=>{if(active)setFailed(true)})
   .finally(()=>{if(active)setLoading(false)})
  return()=>{active=false}
 },[refresh])
 return <div className="inspector-model-picker">
  <label>Available UCloud inspector models<select disabled={loading||failed||!models.length} value={models.includes(value)?value:''} onChange={e=>{if(e.target.value)onChange(e.target.value)}}>
   <option value="">{loading?'Loading UCloud models…':'Select a inspector model'}</option>
   {models.map(model=><option key={model} value={model}>{model}</option>)}
  </select></label>
  <button type="button" disabled={loading} onClick={()=>setRefresh(n=>n+1)}>Refresh inspector models</button>
  {failed&&<p role="status">Could not load UCloud models. Check the server’s inference connection, retry, or enter a model identifier below.</p>}
  {!loading&&!failed&&!models.length&&<p role="status">UCloud returned no models. You can enter a model identifier below.</p>}
  <label>Inspector model identifier<input required value={value} placeholder="zai-org/GLM-5.3" onChange={e=>onChange(e.target.value)}/></label>
  <p className="field-help">Choose an available model or enter its exact identifier. Preview and save to create a new evaluation revision; existing runs retain their inspector.</p>
 </div>
}
