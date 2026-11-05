import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react'
import { it, expect, vi, afterEach } from 'vitest'
import InspectorModelPicker from './InspectorModelPicker'
import { api } from './api'
vi.mock('./api', () => ({ api: vi.fn() }))
afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})
it('discovers UCloud inspectors without replacing the saved choice, and allows selection', async () => {
  vi.mocked(api).mockResolvedValue(['provider/second', 'provider/first'])
  const change = vi.fn()
  render(<InspectorModelPicker value="saved/model" onChange={change} />)
  await screen.findByRole('option', { name: 'provider/second' })
  expect(change).not.toHaveBeenCalled()
  expect(screen.getByLabelText('Inspector model identifier')).toHaveValue('saved/model')
  fireEvent.change(screen.getByLabelText('Available UCloud inspector models'), {
    target: { value: 'provider/second' },
  })
  expect(change).toHaveBeenCalledWith('provider/second')
  expect(api).toHaveBeenCalledWith('/providers/ucloud/models')
})
it('retains manual entry when discovery fails and supports retry', async () => {
  vi.mocked(api)
    .mockRejectedValueOnce(new Error('unavailable'))
    .mockResolvedValueOnce(['restored/model'])
  const change = vi.fn()
  render(<InspectorModelPicker value="saved/model" onChange={change} />)
  await screen.findByRole('status')
  fireEvent.change(screen.getByLabelText('Inspector model identifier'), {
    target: { value: 'manual/model' },
  })
  expect(change).toHaveBeenCalledWith('manual/model')
  fireEvent.click(screen.getByRole('button', { name: 'Refresh inspector models' }))
  await waitFor(() =>
    expect(screen.getByRole('option', { name: 'restored/model' })).toBeInTheDocument(),
  )
})
