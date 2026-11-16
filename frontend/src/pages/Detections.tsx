'use client'

import { useState } from 'react'
import { Search, Shield, Download, RefreshCw } from 'lucide-react'
import { clsx } from 'clsx'
import { toast } from 'react-hot-toast'
import { useInferenceJobs } from '@/hooks/useDashboard'
import { api } from '@/services/api'
import { format } from 'date-fns'
import { useNavigate } from 'react-router-dom'

interface InferenceJob {
  id: string
  name: string
  model_id: string
  status: string
  total_records?: number
  benign_count?: number
  malicious_count?: number
  created_at: string
}

export function Detections() {
  const { data: jobs, isLoading, refetch } = useInferenceJobs()
  const navigate = useNavigate()

  const [search, setSearch] = useState('')
  const [filterStatus, setFilterStatus] = useState('all')

  const totalAnalyzed = jobs?.reduce((sum: number, j: InferenceJob) => sum + (j.total_records || 0), 0) || 0
  const totalBenign = jobs?.reduce((sum: number, j: InferenceJob) => sum + (j.benign_count || 0), 0) || 0
  const totalMalicious = jobs?.reduce((sum: number, j: InferenceJob) => sum + (j.malicious_count || 0), 0) || 0
  const threatRate = totalAnalyzed > 0 ? ((totalMalicious / totalAnalyzed) * 100).toFixed(1) : '0'

  const filteredJobs = jobs?.filter((job: InferenceJob) => {
    const matchesSearch = job.name.toLowerCase().includes(search.toLowerCase())
    const matchesStatus = filterStatus === 'all' || job.status === filterStatus
    return matchesSearch && matchesStatus
  }) || []

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-500/20 text-green-400'
      case 'running': return 'bg-blue-500/20 text-blue-400'
      case 'pending': return 'bg-yellow-500/20 text-yellow-400'
      case 'failed': return 'bg-red-500/20 text-red-400'
      default: return 'bg-dark-700 text-dark-400'
    }
  }

  return (
    <div className="space-y-6 animate-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-50">Detections</h1>
          <p className="text-dark-400 mt-1">Threat detection history and analysis</p>
        </div>
        <button
          onClick={() => void refetch()}
          className="px-4 py-2 bg-primary-500 text-dark-950 rounded-lg font-medium hover:bg-primary-400 flex items-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="p-4 rounded-xl border border-dark-700 bg-dark-900/50">
          <p className="text-sm text-dark-400">Total Analyzed</p>
          <p className="text-2xl font-bold text-dark-50 mt-1">{totalAnalyzed.toLocaleString()}</p>
        </div>
        <div className="p-4 rounded-xl border border-dark-700 bg-dark-900/50">
          <p className="text-sm text-dark-400">Benign</p>
          <p className="text-2xl font-bold text-green-400 mt-1">{totalBenign.toLocaleString()}</p>
        </div>
        <div className="p-4 rounded-xl border border-dark-700 bg-dark-900/50">
          <p className="text-sm text-dark-400">Malicious</p>
          <p className="text-2xl font-bold text-red-400 mt-1">{totalMalicious.toLocaleString()}</p>
        </div>
        <div className="p-4 rounded-xl border border-dark-700 bg-dark-900/50">
          <p className="text-sm text-dark-400">Threat Rate</p>
          <p className="text-2xl font-bold text-dark-50 mt-1">{threatRate}%</p>
        </div>
      </div>

      <div className="p-6 rounded-xl border border-dark-700 bg-dark-900/50">
        <div className="flex gap-4 mb-6">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-400" />
            <input
              type="text"
              placeholder="Search detections..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 placeholder-dark-400 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
            />
          </div>
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
          >
            <option value="all">All Status</option>
            <option value="completed">Completed</option>
            <option value="running">Running</option>
            <option value="pending">Pending</option>
            <option value="failed">Failed</option>
          </select>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-dark-700">
                <th className="pb-3 font-medium text-dark-300">Timestamp</th>
                <th className="pb-3 font-medium text-dark-300">Job Name</th>
                <th className="pb-3 font-medium text-dark-300">Model</th>
                <th className="pb-3 font-medium text-dark-300">Status</th>
                <th className="pb-3 font-medium text-dark-300">Prediction</th>
                <th className="pb-3 font-medium text-dark-300">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-800">
              {isLoading ? (
                <tr>
                  <td className="py-4 text-dark-400 text-center" colSpan={6}>
                    <div className="flex items-center justify-center gap-2">
                      <RefreshCw className="w-5 h-5 animate-spin" />
                      Loading...
                    </div>
                  </td>
                </tr>
              ) : filteredJobs.length > 0 ? (
                filteredJobs.map((job: InferenceJob) => (
                  <tr key={job.id} className="hover:bg-dark-800/50 transition-colors">
                    <td className="py-4 text-dark-400 text-sm font-mono">
                      {job.created_at ? format(new Date(job.created_at), 'MMM d, yyyy HH:mm') : '-'}
                    </td>
                    <td className="py-4 text-dark-50">{job.name}</td>
                    <td className="py-4 text-dark-400 font-mono text-sm">{job.model_id.slice(0, 8)}...</td>
                    <td className="py-4">
                      <span className={clsx('px-2 py-1 text-xs rounded-full', getStatusColor(job.status))}>
                        {job.status}
                      </span>
                    </td>
                    <td className="py-4 text-dark-300">
                      {job.malicious_count?.toLocaleString() ?? '-'} malicious / {job.benign_count?.toLocaleString() ?? '-'} benign
                    </td>
                    <td className="py-4">
                      <div className="flex items-center gap-2">
                        {job.status === 'completed' && (
                          <button
                            className="px-3 py-1.5 text-sm bg-primary-500/10 text-primary-400 rounded-lg hover:bg-primary-500/20 flex items-center gap-1 transition-colors"
                            onClick={() => navigate('/inference', { state: { viewJobId: job.id } })}
                          >
                            <Shield className="w-4 h-4" />
                            View
                          </button>
                        )}
                        {job.status === 'completed' && (
                          <button
                            className="px-3 py-1.5 text-sm bg-green-500/10 text-green-400 rounded-lg hover:bg-green-500/20 flex items-center gap-1 transition-colors"
                            onClick={() => {
                              const csv = [
                                ['Job ID', 'Name', 'Model ID', 'Status', 'Total Records', 'Benign', 'Malicious', 'Created At'],
                                [job.id, job.name, job.model_id, job.status, job.total_records ?? 0, job.benign_count ?? 0, job.malicious_count ?? 0, job.created_at]
                              ]
                                .map((r) => r.map(String).join(','))
                                .join('\n')
                              const blob = new Blob([csv], { type: 'text/csv' })
                              const url = URL.createObjectURL(blob)
                              const a = document.createElement('a')
                              a.href = url
                              a.download = `detection_${job.id.slice(0, 8)}_${Date.now()}.csv`
                              a.click()
                              URL.revokeObjectURL(url)
                            }}
                          >
                            <Download className="w-4 h-4" />
                            Export
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td className="py-4 text-dark-400 text-center" colSpan={6}>No detections yet</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
