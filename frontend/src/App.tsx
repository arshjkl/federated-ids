import { Routes, Route, Navigate } from 'react-router-dom'
import { Shield, Network, Brain, BarChart3, Settings as SettingsIcon, AlertTriangle } from 'lucide-react'
import { Sidebar } from '@/components/layout/Sidebar'
import { Header } from '@/components/layout/Header'
import { Dashboard } from '@/pages/Dashboard'
import { Organisations } from '@/pages/Organisations'
import { Experiments } from '@/pages/Experiments'
import { Inference } from '@/pages/Inference'
import { Detections } from '@/pages/Detections'
import { Privacy } from '@/pages/Privacy'
import { Settings } from '@/pages/Settings'
import { ThemeProvider } from '@/components/providers/ThemeProvider'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

const navigation = [
  { name: 'Dashboard', href: '/', icon: BarChart3 },
  { name: 'Organisations', href: '/organisations', icon: Network },
  { name: 'Experiments', href: '/experiments', icon: Brain },
  { name: 'Inference', href: '/inference', icon: Shield },
  { name: 'Detections', href: '/detections', icon: AlertTriangle },
  { name: 'Privacy Analytics', href: '/privacy', icon: SettingsIcon },
]

function AppLayout() {
  return (
    <div className="flex h-screen bg-dark-950 text-dark-50 overflow-hidden">
      <Sidebar navigation={navigation} />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto p-6">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/organisations" element={<Organisations />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/inference" element={<Inference />} />
            <Route path="/detections" element={<Detections />} />
            <Route path="/privacy" element={<Privacy />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AppLayout />
      </ThemeProvider>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  )
}