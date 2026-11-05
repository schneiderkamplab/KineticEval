import {useState} from 'react'
const human=(key:string)=>key.replaceAll('_',' ').replace(/^./,c=>c.toUpperCase())
export function ReadableValue({value}:{value:any}){
 if(value===null||value===undefined)return <span className="muted">Not specified</span>
 if(typeof value==='boolean')return <span>{value?'Yes':'No'}</span>
 if(Array.isArray(value))return value.length?<ol>{value.map((v,i)=><li key={i}><ReadableValue value={v}/></li>)}</ol>:<span className="muted">None</span>
 if(typeof value==='object')return <dl>{Object.entries(value).map(([key,v])=><div key={key}><dt>{human(key)}</dt><dd><ReadableValue value={v}/></dd></div>)}</dl>
 return <span style={{whiteSpace:'pre-wrap',overflowWrap:'anywhere'}}>{String(value)}</span>
}
export function StructuredEditor({value,onChange,label='Settings'}:{value:any,onChange:(value:any)=>void,label?:string}){
 const [key,setKey]=useState('')
 const [kind,setKind]=useState('text')
 const initial=()=>kind==='number'?0:kind==='boolean'?false:kind==='object'?{}:kind==='list'?[]:kind==='null'?null:''
 if(Array.isArray(value))return <fieldset><legend>{label}</legend>{value.map((v,i)=><div key={i}><StructuredEditor label={`${label} ${i+1}`} value={v} onChange={next=>onChange(value.map((old,j)=>j===i?next:old))}/><button type="button" onClick={()=>onChange(value.filter((_,j)=>i!==j))}>Remove item {i+1}</button></div>)}<label>New item type<select value={kind} onChange={e=>setKind(e.target.value)}>{['text','number','boolean','object','list','null'].map(t=><option key={t}>{t}</option>)}</select></label><button type="button" onClick={()=>onChange([...value,initial()])}>Add item</button></fieldset>
 if(value!==null&&typeof value==='object')return <fieldset><legend>{label}</legend>{Object.entries(value).map(([k,v])=><div key={k}><StructuredEditor label={human(k)} value={v} onChange={next=>onChange({...value,[k]:next})}/><button type="button" className="link" onClick={()=>{const next={...value};delete next[k];onChange(next)}}>Remove {human(k)}</button></div>)}<div className="form-columns"><label>New setting name<input value={key} onChange={e=>setKey(e.target.value)}/></label><label>Value type<select value={kind} onChange={e=>setKind(e.target.value)}>{['text','number','boolean','object','list','null'].map(t=><option key={t}>{t}</option>)}</select></label></div><button type="button" disabled={!key.trim()||key in value} onClick={()=>{onChange({...value,[key.trim()]:initial()});setKey('')}}>Add setting</button></fieldset>
 if(typeof value==='boolean')return <label className="check"><input type="checkbox" checked={value} onChange={e=>onChange(e.target.checked)}/>{label}</label>
 if(value===null)return <label>{label}<select value="null" onChange={e=>onChange(e.target.value==='number'?0:e.target.value==='boolean'?false:'')}><option value="null">Not specified</option><option value="text">Set text</option><option value="number">Set number</option><option value="boolean">Set yes/no</option></select></label>
 return <label>{label}{typeof value==='number'?<input type="number" value={value} onChange={e=>onChange(Number(e.target.value))}/>:<textarea rows={String(value).includes('\n')?4:1} value={value} onChange={e=>onChange(e.target.value)}/>}</label>
}
export function AdvancedJson({value,onChange}:{value:any,onChange?:(value:any)=>void}){
 const [text,setText]=useState(''),[error,setError]=useState('')
 return <details onToggle={e=>{if(e.currentTarget.open)setText(JSON.stringify(value,null,2))}}><summary>Advanced · {onChange?'View/edit':'View'} JSON</summary>{onChange?<><textarea aria-label="Advanced JSON" rows={12} value={text} onChange={e=>setText(e.target.value)}/>{error&&<p role="alert">{error}</p>}<button onClick={()=>{try{onChange(JSON.parse(text));setError('')}catch(e){setError(String(e))}}}>Apply JSON to fields</button></>:<pre>{JSON.stringify(value,null,2)}</pre>}</details>
}
