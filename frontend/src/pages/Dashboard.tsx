'use client'

import { useNavigate } from 'react-router-dom'
import {
  Users,
  Brain,
  Shield,
  AlertTriangle,
  Activity,
  Loader2,
  ChevronRight,
} from 'lucide-react'
import { clsx } from 'clsx'
import { useDashboardStats } from '@/hooks/useDashboard'

const statConfig = [
  { name: 'Organisations', key: 'total_organisations', icon: Users, color: 'text-blue-400', bg: 'bg-blue-500/10', href: '/organisations' },
  { name: 'Active Experiments', key: 'active_experiments', icon: Brain, color: 'text-purple-400', bg: 'bg-purple-500/10', href: '/experiments' },
  { name: 'Inference Jobs', key: 'total_inference_jobs', icon: Shield, color: 'text-orange-400', bg: 'bg-orange-500/10', href: '/inference' },
  { name: 'Total Detections', key: 'total_detections', icon: AlertTriangle, color: 'text-red-400', bg: 'bg-red-500/10', href: '/detections' },
  { name: 'Malicious Detected', key: 'malicious_detections', icon: Activity, color: 'text-pink-400', bg: 'bg-pink-500/10', href: '/detections' },
]

const actionButtons = [
  { name: 'Add Organisation', description: 'Register a new organisation', href: '/organisations', icon: Users },
  { name: 'Create Experiment', description: 'Start federated training', href: '/experiments', icon: Brain },
  { name: 'Run Inference', description: 'Test model on dataset', href: '/inference', icon: Shield },
  { name: 'View Detections', description: 'Analyze threat alerts', href: '/detections', icon: AlertTriangle },
]

export function Dashboard() {
  const navigate = useNavigate()
  const { data: stats, isLoading: statsLoading, error: statsError, refetch } = useDashboardStats()

  const handleNavigate = (href: string) => {
    navigate(href)
  }

  if (statsLoading) {
    return (
      <div className="space-y-6 animate-in">
        <div>
          <h1 className="text-2xl font-bold text-dark-50">Dashboard</h1>
          <p className="text-dark-400 mt-1">Federated Learning IDS Platform Overview</p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
          {statConfig.map((stat, i) => {
            const Icon = stat.icon
            return (
              <div
                key={stat.name}
                className={clsx(
                  'p-5 rounded-xl border border-dark-700 bg-dark-900/50 transition-all hover:border-dark-600 animate-pulse',
                  'animate-in'
                )}
                style={{ animationDelay: `${i * 50}ms` }}
              >
                <div className="flex items-center justify-between">
                  <div className={clsx('p-2 rounded-lg', stat.bg)}>
                    <Icon className={clsx('w-5 h-5', stat.color)} />
                  </div>
                  <div className="h-8 w-16 bg-dark-700 rounded animate-pulse" />
                </div>
                <div className="mt-3">
                  <p className="text-sm text-dark-400">{stat.name}</p>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    )
  }

  if (statsError) {
    return (
      <div className="space-y-6 animate-in">
        <div>
          <h1 className="text-2xl font-bold text-dark-50">Dashboard</h1>
          <p className="text-dark-400 mt-1">Federated Learning IDS Platform Overview</p>
        </div>
        <div className="p-6 rounded-xl border border-red-500/50 bg-red-500/10">
          <p className="text-red-400">Failed to load dashboard stats. Please check the backend connection.</p>
          <button
            onClick={() => refetch()}
            className="mt-4 px-4 py-2 bg-primary-500 text-dark-950 rounded-lg font-medium hover:bg-primary-400"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-50">Dashboard</h1>
          <p className="text-dark-400 mt-1">Federated Learning IDS Platform Overview</p>
        </div>
        <button
          onClick={() => refetch()}
          className="px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg flex items-center gap-2 text-dark-300 hover:bg-dark-700 transition-colors"
        >
          <Loader2 className="w-4 h-4" />
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        {statConfig.map((stat, i) => {
          const Icon = stat.icon
          const value = stats?.[stat.key] ?? 0
          return (
            <div
              key={stat.name}
              className={clsx(
                'p-5 rounded-xl border border-dark-700 bg-dark-900/50 transition-all hover:border-dark-600 cursor-pointer',
                'animate-in'
              )}
              style={{ animationDelay: `${i * 50}ms` }}
              onClick={() => handleNavigate(stat.href)}
            >
              <div className="flex items-center justify-between">
                <div className={clsx('p-2 rounded-lg', stat.bg)}>
                  <Icon className={clsx('w-5 h-5', stat.color)} />
                </div>
                <span className="text-2xl font-bold text-dark-50">{value.toLocaleString()}</span>
              </div>
              <div className="mt-3">
                <p className="text-sm text-dark-400">{stat.name}</p>
              </div>
            </div>
          )
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="p-6 rounded-xl border border-dark-700 bg-dark-900/50">
          <h3 className="text-lg font-semibold text-dark-50 mb-4">Quick Actions</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {actionButtons.map((action, i) => {
              const Icon = action.icon
              return (
                <button
                  key={action.name}
                  onClick={() => handleNavigate(action.href)}
                  className={clsx(
                    'p-4 rounded-lg border border-dark-700 bg-dark-800/50 hover:border-primary-500/50 hover:bg-primary-500/5 transition-all text-left flex items-center gap-3',
                    'animate-in'
                  )}
                  style={{ animationDelay: `${i * 50}ms` }}
                >
                  <div className="p-2 rounded-lg bg-primary-500/10 text-primary-400">
                    <Icon className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="font-medium text-dark-50">{action.name}</p>
                    <p className="text-sm text-dark-400 mt-1">{action.description}</p>
                  </div>
                  <ChevronRight className="w-4 h-4 text-dark-500 ml-auto" />
                </button>
              )
            })}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="p-6 rounded-xl border border-dark-700 bg-dark-900/50">
          <h3 className="text-lg font-semibold text-dark-50 mb-4">System Status</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-dark-800/50 rounded-lg">
              <span className="text-dark-300">Backend API</span>
              <span className="flex items-center gap-2 text-green-400">
                <span className="w-2 h-2 rounded-full bg-green-400" />
                Connected
              </span>
            </div>
            <div className="flex items-center justify-between p-3 bg-dark-800/50 rounded-lg">
              <span className="text-dark-300">Database</span>
              <span className="flex items-center gap-2 text-green-400">
                <span className="w-2 h-2 rounded-full bg-green-400" />
                Connected (SQLite)
              </span>
            </div>
          </div>
        </div>

        <div className="p-6 rounded-xl border border-dark-700 bg-dark-900/50">
          <h3 className="text-lg font-semibold text-dark-50 mb-4">Recent Activity</h3>
          <div className="space-y-3">
            <p className="text-dark-400 text-center py-8">No activity yet - create an experiment or run inference to see activity</p>
          </div>
        </div>
      </div>
    </div>
  )
}