import {useEffect,useRef,type ReactNode} from 'react'
import {X} from 'lucide-react'
export default function Modal({title,close,children,wide=false}:{title:string,close:()=>void,children:ReactNode,wide?:boolean}){
 const container=useRef<HTMLElement>(null)
 useEffect(()=>{const previous=document.activeElement as HTMLElement|null;container.current?.focus();return()=>previous?.focus()},[])
 return <div className="scrim" onClick={close}><section ref={container} tabIndex={-1} className={'modal'+(wide?' modal-wide':'')} role="dialog" aria-modal="true" aria-label={title} onClick={e=>e.stopPropagation()} onKeyDown={e=>{if(e.key==='Escape'){e.stopPropagation();close()}if(e.key==='Tab'){const fields=[...(container.current?.querySelectorAll<HTMLElement>('button:not(:disabled),input:not(:disabled),select:not(:disabled),textarea:not(:disabled),summary,[tabindex="0"]')||[])].filter(el=>el.getClientRects().length);const first=fields[0],last=fields.at(-1);if(e.shiftKey&&(document.activeElement===first||document.activeElement===container.current)){e.preventDefault();last?.focus()}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first?.focus()}}}}><header><h2>{title}</h2><button aria-label="Close" className="icon" onClick={close}><X size={20}/></button></header>{children}</section></div>
}
