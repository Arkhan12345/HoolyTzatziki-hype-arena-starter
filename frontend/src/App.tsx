import { lazy, Suspense } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Layout } from '@/components/Layout'
import Dashboard from '@/pages/Dashboard'
import { DebugProvider } from '@/state/debug'
import { RunProvider } from '@/state/run'
import { ThemeProvider } from '@/lib/theme'

const VerifyPage = lazy(() => import('@/pages/Verify'))
const ExplorerPage = lazy(() => import('@/pages/Explorer'))
const GeneratePage = lazy(() => import('@/pages/Generate'))
const ArchitecturePage = lazy(() => import('@/pages/Architecture'))
const MetricsPage = lazy(() => import('@/pages/Metrics'))
const PitchPage = lazy(() => import('@/pages/Pitch'))
const OutputPage = lazy(() => import('@/pages/Output'))
const ValidatePage = lazy(() => import('@/pages/Validate'))

function Loading() {
  return (
    <div className="space-y-4 p-2" aria-busy="true" aria-label="Loading">
      <div className="skeleton h-8 w-64" />
      <div className="skeleton h-40" />
    </div>
  )
}

export default function App() {
  return (
    <ThemeProvider>
      <DebugProvider>
      <BrowserRouter>
        <RunProvider>
          <Suspense fallback={<Loading />}>
            <Routes>
              <Route element={<Layout />}>
                <Route index element={<Dashboard />} />
                <Route path="run" element={<Dashboard />} />
                <Route path="verify" element={<VerifyPage />} />
                <Route path="explorer" element={<ExplorerPage />} />
                <Route path="generate" element={<GeneratePage />} />
                <Route path="output" element={<OutputPage />} />
                <Route path="architecture" element={<ArchitecturePage />} />
                <Route path="metrics" element={<MetricsPage />} />
                <Route path="validate" element={<ValidatePage />} />
                <Route path="pitch" element={<PitchPage />} />
                <Route path="*" element={<Dashboard />} />
              </Route>
            </Routes>
          </Suspense>
        </RunProvider>
      </BrowserRouter>
      </DebugProvider>
    </ThemeProvider>
  )
}
