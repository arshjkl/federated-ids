'use client'

import { useState } from 'react'
import { Plus, Search, Filter, Trash2, Edit2, Eye } from 'lucide-react'
import { clsx } from 'clsx'
import { toast } from 'react-hot-toast'
import { useModels } from '@/hooks/useDashboard'
import { api } from '@/services/api'

interface GlobalModel {
  id: string
  name: string
  version: string
  algorithm: string
  architecture: string
  status: string
  metrics?: Record<string, number>
  training_rounds: number
  participating_clients: number
  created_at: string
}

interface AlgorithmOption {
  value: string
  label: string
}

const ALGORITHMS: AlgorithmOption[] = [
  { value: 'fedavg', label: 'FedAvg' },
  { value: 'fedprox', label: 'FedProx' },
  { value: 'dp_fedavg', label: 'DP-FedAvg' },
  { value: 'dp_fedprox', label: 'DP-FedProx' },
  { value: 'centralized', label: 'Centralized' },
]

export function Models() {
  const { data: models, isLoading, refetch } = useModels()

  const [showCreateModal, setShowCreateModal] = useState(false)
  const [creating, setCreating] = useState(false)
  const [createForm, setCreateForm] = useState({
    version: '',
    name: '',
    description: '',
    algorithm: 'fedavg',
    architecture: 'MLPIDS',
    checkpoint_path: '',
    training_rounds: 20,
    participating_clients: 1,
  })

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault()
    setCreating(true)
    try {
      await api.createModel(createForm)
      toast.success('Model registered')
      setShowCreateModal(false)
      setCreateForm({
        version: '',
        name: '',
        description: '',
        algorithm: 'fedavg',
        architecture: 'MLPIDS',
        checkpoint_path: '',
        training_rounds: 20,
        participating_clients: 1,
      })
      refetch()
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to register model')
    } finally {
      setCreating(false)
    }
  }

  const handleDelete = async (id: string) => {
    if (!window.confirm('Delete this model?')) return
    try {
      await api.deleteModel(id)
      toast.success('Model deleted')
      refetch()
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to delete model')
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500/20 text-green-400'
      case 'training': return 'bg-blue-500/20 text-blue-400'
      case 'validating': return 'bg-yellow-500/20 text-yellow-400'
      case 'failed': return 'bg-red-500/20 text-red-400'
      case 'archived': return 'bg-gray-500/20 text-gray-400'
      default: return 'bg-dark-700 text-dark-400'
    }
  }

  const accuracy = (m: GlobalModel) => {
    return m.metrics?.accuracy ?? m.metrics?.test_accuracy ?? '-'
  }

  return (
    <div className="space-y-6 animate-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-50">Global Models</h1>
          <p className="text-dark-400 mt-1">Versioned global model registry</p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-primary-500 text-dark-950 rounded-lg font-medium hover:bg-primary-400 flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          Register Model
        </button>
      </div>

      <div className="p-6 rounded-xl border border-dark-700 bg-dark-900/50">
        <div className="flex gap-4 mb-6">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-400" />
            <input
              type="text"
              placeholder="Search models..."
              className="w-full pl-10 pr-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 placeholder-dark-400 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
            />
          </div>
          <button className="px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg flex items-center gap-2 text-dark-300 hover:bg-dark-700">
            <Filter className="w-4 h-4" />
            Filter
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-dark-700">
                <th className="pb-3 font-medium text-dark-300">Version</th>
                <th className="pb-3 font-medium text-dark-300">Name</th>
                <th className="pb-3 font-medium text-dark-300">Algorithm</th>
                <th className="pb-3 font-medium text-dark-300">Architecture</th>
                <th className="pb-3 font-medium text-dark-300">Status</th>
                <th className="pb-3 font-medium text-dark-300">Accuracy</th>
                <th className="pb-3 font-medium text-dark-300">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-800">
              {isLoading ? (
                <tr>
                  <td className="py-4 text-dark-400 text-center" colSpan={7}>
                    <div className="flex items-center justify-center gap-2">
                      <Eye className="w-5 h-5 animate-spin" />
                      Loading...
                    </div>
                  </td>
                </tr>
              ) : models && models.length > 0 ? (
                (models as GlobalModel[]).map((m) => (
                  <tr key={m.id} className="hover:bg-dark-800/50 transition-colors">
                    <td className="py-4 text-dark-400 font-mono text-sm">{m.version}</td>
                    <td className="py-4 text-dark-50">{m.name}</td>
                    <td className="py-4 text-dark-400 text-capitalize">{m.algorithm}</td>
                    <td className="py-4 text-dark-400">{m.architecture}</td>
                    <td className="py-4">
                      <span className={clsx('px-2 py-1 text-xs rounded-full', getStatusColor(m.status))}>
                        {m.status}
                      </span>
                    </td>
                    <td className="py-4 text-dark-300 font-mono">
                      {typeof accuracy(m) === 'number' ? (accuracy(m) as number * 100).toFixed(1) + '%' : accuracy(m)}
                    </td>
                    <td className="py-4">
                      <div className="flex items-center gap-2">
                        <button
                          className="px-3 py-1.5 text-sm bg-dark-700 text-dark-300 rounded-lg hover:bg-dark-600 flex items-center gap-1 transition-colors"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button
                          className="px-3 py-1.5 text-sm bg-primary-500/10 text-primary-400 rounded-lg hover:bg-primary-500/20 flex items-center gap-1 transition-colors"
                          onClick={() => toast('Edit coming soon')}
                        >
                          <Edit2 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleDelete(m.id)}
                          className="px-3 py-1.5 text-sm bg-red-500/10 text-red-400 rounded-lg hover:bg-red-500/20 flex items-center gap-1 transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td className="py-4 text-dark-400 text-center" colSpan={7}>No models registered yet</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Create Model Modal */}
      <div
        className={clsx('fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm', showCreateModal ? '' : 'hidden')}
      >
        <div className="w-full max-w-md bg-dark-900 border border-dark-700 rounded-xl overflow-hidden">
          <div className="p-4 border-b border-dark-700 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-dark-50">Register Model</h2>
            <button
              onClick={() => setShowCreateModal(false)}
              className="p-2 rounded-lg text-dark-400 hover:text-dark-500 hover:bg-dark-800 transition-colors"
            >
              <Eye className="w-5 h-5" />
            </button>
          </div>
          <form onSubmit={handleCreate} className="p-6 space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-dark-300">Version</label>
              <input
                type="text"
                value={createForm.version}
                onChange={(e) => setCreateForm({ ...createForm, version: e.target.value })}
                placeholder="e.g. centralized-baseline-v1"
                className="w-full px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                required
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-dark-300">Name</label>
              <input
                type="text"
                value={createForm.name}
                onChange={(e) => setCreateForm({ ...createForm, name: e.target.value })}
                placeholder="Model display name"
                className="w-full px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                required
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-dark-300">Description</label>
              <input
                type="text"
                value={createForm.description}
                onChange={(e) => setCreateForm({ ...createForm, description: e.target.value })}
                placeholder="Optional description"
                className="w-full px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-dark-300">Algorithm</label>
              <select
                value={createForm.algorithm}
                onChange={(e) => setCreateForm({ ...createForm, algorithm: e.target.value })}
                className="w-full px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
              >
                {ALGORITHMS.map((a) => (
                  <option key={a.value} value={a.value}>
                    {a.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-dark-300">Architecture</label>
              <input
                type="text"
                value={createForm.architecture}
                onChange={(e) => setCreateForm({ ...createForm, architecture: e.target.value })}
                placeholder="e.g. MLPIDS"
                className="w-full px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                required
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-dark-300">Checkpoint Path</label>
              <input
                type="text"
                value={createForm.checkpoint_path}
                onChange={(e) => setCreateForm({ ...createForm, checkpoint_path: e.target.value })}
                placeholder="/path/to/model.pth"
                className="w-full px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                required
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-dark-300">Training Rounds</label>
                <input
                  type="number"
                  value={createForm.training_rounds}
                  onChange={(e) => setCreateForm({ ...createForm, training_rounds: parseInt(e.target.value) || 0 })}
                  className="w-full px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-dark-300">Participating Clients</label>
                <input
                  type="number"
                  value={createForm.participating_clients}
                  onChange={(e) => setCreateForm({ ...createForm, participating_clients: parseInt(e.target.value) || 0 })}
                  className="w-full px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                />
              </div>
            </div>
            <div className="flex justify-end gap-3 pt-4 border-t border-dark-700">
              <button
                type="button"
                onClick={() => setShowCreateModal(false)}
                className="px-4 py-2 bg-dark-800 text-dark-300 rounded-lg hover:bg-dark-700"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={creating}
                className="px-4 py-2 bg-primary-500 text-dark-950 rounded-lg font-medium hover:bg-primary-400 disabled:opacity-50"
              >
                {creating ? 'Registering...' : 'Register Model'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}
