import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { Coverage } from './App'
describe('coverage', () => {
  it('makes a partial denominator explicit', () => {
    render(
      <Coverage
        summary={{
          model: 'test',
          scored: 3,
          total: 10,
          metrics: { accuracy: 1 },
          full_evaluation: false,
          reused: 2,
        }}
      />,
    )
    expect(screen.getByText('3 / 10 scored')).toBeInTheDocument()
    expect(screen.getByText(/Partial coverage/)).toBeInTheDocument()
    expect(screen.getByText('100%')).toBeInTheDocument()
  })
})
