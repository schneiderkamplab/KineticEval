import {useEffect,useState} from 'react'
import {api} from './api'
type Row=Record<string,any>
const labels:Record<string,string>={generation:'Generating responses',generating:'Generating responses · model ready',unloading_model:'Unloading previous model',preparing_model:'Preparing model · downloading missing files / loading weights',scoring:'Scoring',diagnostic:'Heuristic checks',inspector:'LLM inspection',metrics:'Computing metrics',investigation:'Investigating patterns',report:'Writing report'}
function Counts({value,label}:{value:Row;label:string}){
 const counts=value.counts,done=counts.succeeded||0,failed=counts.failed||0,running=counts.running||0,pending=counts.pending||0,waiting=counts.waiting||0
 const complete=done===value.total&&value.total>0
 const status=complete?'Completed':[
  failed?failed+' failed':'',running?running+' running':'',pending?pending+' queued for a worker':'',waiting?waiting+' awaiting earlier stages':''
 ].filter(Boolean).join(' · ')||'Not started'
 return <div className="stage-progress"><div><span>{label}</span><strong>{done.toLocaleString()} / {value.total.toLocaleString()}</strong></div><progress aria-label={label} max={value.total||1} value={done}/><small className={failed?'progress-failed':''}>{status}{value.dynamic&&!complete?' · scheduled so far':''}</small></div>
}
export default function RunProgress({id,compact=false,status}:{id:string;compact?:boolean;status?:string}){
 const [data,setData]=useState<Row|null>(null),[error,setError]=useState(false)
 useEffect(()=>{
  let active=true,timer:ReturnType<typeof setTimeout>
  async function poll(){try{const value=await api('/runs/'+id+'/progress');if(active){setData(value);setError(false);if(value.status==='active')timer=setTimeout(poll,3000)}}catch{if(active){setError(true);timer=setTimeout(poll,5000)}}}
  void poll();return()=>{active=false;clearTimeout(timer)}
 },[id,status])
 if(!data)return <p role="status">{error?'Progress unavailable; retrying…':'Loading progress…'}</p>
 return <section className={compact?'run-progress compact':'panel run-progress'} aria-label="Run progress">
 {!compact&&<h2>Run progress</h2>}
 <p role="status">{data.status!=='active'?'Run '+data.status.replaceAll('_',' '):data.activities.length?data.activities.map((a:Row)=>(a.stale?'Worker lease expired; waiting for recovery':labels[a.phase]||a.phase)+(a.model?' — '+a.model:'')).join(' · '):'Waiting for a worker or the next stage'}</p>
 {error&&<small role="alert">Connection interrupted. Showing last known progress; retrying…</small>}
 {data.stages.map((stage:Row)=><Counts key={stage.kind} value={stage} label={labels[stage.kind]||stage.kind}/>)}
 {!compact&&<><p className="field-help">{data.reused} reused responses. Inspector, investigation, and report work is scheduled as results become available.</p><details><summary>Generation by model</summary>{data.models.map((model:Row)=><Counts key={model.name} value={model} label={model.name}/>)}</details></>}
 </section>
}
