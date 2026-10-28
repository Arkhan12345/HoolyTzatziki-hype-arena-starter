import { lazy, Suspense } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Layout } from '@/components/Layout'
import RunPage from '@/pages/Run'
import { RunProvider } from '@/state/run'

const ArchitecturePage = lazy(() => import('@/pages/Architecture'))
const MetricsPage = lazy(() => import('@/pages/Metrics'))
const PitchPage = lazy(() => import('@/pages/Pitch'))

export default function App() {
  return (
    <BrowserRouter>
      <RunProvider>
        <Suspense fallback={<div className="grid h-64 place-items-center text-ink-muted">Loading…</div>}>
          <Routes>
            <Route element={<Layout />}>
              <Route index element={<RunPage />} />
              <Route path="architecture" element={<ArchitecturePage />} />
              <Route path="metrics" element={<MetricsPage />} />
              <Route path="pitch" element={<PitchPage />} />
              <Route path="*" element={<RunPage />} />
            </Route>
          </Routes>
        </Suspense>
      </RunProvider>
    </BrowserRouter>
  )
}
