import { render, screen, waitFor, cleanup } from '@testing-library/react'
import { it, expect, vi, afterEach } from 'vitest'
import RunProgress from './RunProgress'
import { api } from './api'
vi.mock('./api', () => ({ api: vi.fn() }))
afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})
it('shows stage denominators, reused work, failures and model preparation', async () => {
  vi.mocked(api).mockResolvedValue({
    status: 'active',
    reused: 2,
    activities: [{ phase: 'preparing_model', model: 'local/HF' }],
    models: [],
    stages: [
      { kind: 'generation', counts: { succeeded: 4, running: 1, pending: 5 }, total: 10 },
      { kind: 'inspector', counts: { succeeded: 1, failed: 1 }, total: 2, dynamic: true },
    ],
  })
  render(<RunProgress id="run" />)
  await screen.findByText(/Preparing model/)
  expect(screen.getByRole('progressbar', { name: 'Generating responses' })).toHaveAttribute(
    'value',
    '4',
  )
  expect(screen.getByRole('progressbar', { name: 'Generating responses' })).toHaveAttribute(
    'max',
    '10',
  )
  expect(screen.getByText(/1 failed/)).toBeInTheDocument()
  expect(screen.getByText(/2 reused responses/)).toBeInTheDocument()
  expect(screen.getByText(/scheduled so far/)).toBeInTheDocument()
})
it('shows explicit unavailable status instead of fabricated progress', async () => {
  vi.mocked(api).mockRejectedValue(new Error('offline'))
  render(<RunProgress id="run" />)
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Progress unavailable'))
  expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()
})
