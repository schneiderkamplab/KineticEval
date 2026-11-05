import {render,screen,fireEvent} from '@testing-library/react'
import {useState} from 'react'
import {it,expect} from 'vitest'
import {StructuredEditor,AdvancedJson} from './StructuredValue'
it('preserves unknown fields and permits typed edits and JSON round trips',()=>{
 function Fixture(){const [v,setV]=useState({count:2,custom:{enabled:true}});return <><StructuredEditor label="Configuration" value={v} onChange={setV}/><AdvancedJson value={v} onChange={setV}/><output>{JSON.stringify(v)}</output></>}
 render(<Fixture/>);fireEvent.change(screen.getByLabelText('Count'),{target:{value:'3'}})
 expect(screen.getByRole('status')).toHaveTextContent('"custom":{"enabled":true}')
 fireEvent.click(screen.getByText('Advanced · View/edit JSON'))
 fireEvent.change(screen.getByLabelText('Advanced JSON'),{target:{value:'{"count":4,"custom":{"enabled":false}}'}})
 fireEvent.click(screen.getByRole('button',{name:'Apply JSON to fields'}))
 expect(screen.getByLabelText('Count')).toHaveValue(4)
 expect(screen.getByLabelText('Enabled')).not.toBeChecked()
})
