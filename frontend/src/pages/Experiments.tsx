'use client'

import { useState, useEffect } from 'react'
import { Plus, Search, Filter, Trash2, Eye, Play, Pause, Square, RefreshCw, X, Zap, Database, Building2 } from 'lucide-react'
import { clsx } from 'clsx'
import { toast } from 'react-hot-toast'
import { useExperiments, useOrganisations, useDatasets } from '@/hooks/useDashboard'
import { api } from '@/services/api'
import { format } from 'date-fns'

interface Experiment {
  id: string
  name: string
  description?: string
  algorithm: string
  status: string
  total_rounds: number
  current_round: number
  participating_orgs: string[]
  created_at: string
  started_at?: string
  completed_at?: string
  config?: Record<string, any>
}

interface Organisation {
  id: string
  name: string
}

const ALGORITHMS = [
  { value: 'centralized', label: 'Centralized Improved MLP' },
  { value: 'fedavg', label: 'FedAvg' },
  { value: 'fedprox', label: 'FedProx' },
  { value: 'dp_fedavg', label: 'DP-FedAvg' },
  { value: 'dp_fedprox', label: 'DP-FedProx' },
]

export function Experiments() {
  const { data: experiments, isLoading: expsLoading, refetch: refetchExperiments } = useExperiments()
  const { data: organisations } = useOrganisations()
  const { data: registeredDatasets } = useDatasets()

  const [showCreateModal, setShowCreateModal] = useState(false)
  const [creating, setCreating] = useState(false)
  const [viewExperiment, setViewExperiment] = useState<Experiment | null>(null)
  const [createForm, setCreateForm] = useState({
    name: '',
    description: '',
    algorithm: 'fedavg',
    total_rounds: 5,
    sample_file: 'kaggle_cybersecurity_intrusion_2025.csv',
    dataset_id: '',
    target_column: '',
    config: {
      local_epochs: 2,
      batch_size: 128,
      learning_rate: 0.003,
      max_samples_per_client: 100000,
      fedprox_mu: 0.01,
      dp_noise_multiplier: 0.35,
      dp_max_grad_norm: 1.0,
      dp_delta: 0.00001,
      secure_aggregation: false,
      client_sampling_fraction: 1.0,
      checkpoint_interval: 5,
    },
    participating_org_ids: [] as string[],
  })

  useEffect(() => {
    if (organisations && organisations.length > 0 && createForm.participating_org_ids.length === 0) {
      setCreateForm((prev) => ({
        ...prev,
        participating_org_ids: organisations.slice(0, 2).map((o: any) => o.id),
      }))
    }
  }, [organisations])

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault()
    if (createForm.participating_org_ids.length < 1) {
      toast.error('Select at least 1 sponsoring organisation')
      return
    }
    setCreating(true)
    try {
      const payload: any = {
        name: createForm.name,
        description: createForm.description,
        algorithm: createForm.algorithm,
        total_rounds: createForm.total_rounds,
        participating_org_ids: createForm.participating_org_ids,
        dataset_id: createForm.dataset_id || undefined,
        sample_file: createForm.sample_file || undefined,
        target_column: createForm.target_column || undefined,
        config: {
          ...createForm.config,
          dataset_id: createForm.dataset_id || undefined,
          sample_file: createForm.sample_file || undefined,
          target_column: createForm.target_column || undefined,
        },
      }
      await api.createExperiment(payload)
      toast.success('Experiment created. Click Start to train on this dataset!')
      setShowCreateModal(false)
      setCreateForm({
        name: '',
        description: '',
        algorithm: 'fedavg',
        total_rounds: 5,
        sample_file: 'kaggle_cybersecurity_intrusion_2025.csv',
        dataset_id: '',
        target_column: '',
        config: {
          local_epochs: 2,
          batch_size: 128,
          learning_rate: 0.003,
          max_samples_per_client: 100000,
          fedprox_mu: 0.01,
          dp_noise_multiplier: 0.35,
          dp_max_grad_norm: 1.0,
          dp_delta: 0.00001,
          secure_aggregation: false,
          client_sampling_fraction: 1.0,
          checkpoint_interval: 5,
        },
        participating_org_ids: organisations ? organisations.slice(0, 2).map((o: any) => o.id) : [],
      })
      refetchExperiments()
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to create experiment')
    } finally {
      setCreating(false)
    }
  }

  const handleDelete = async (id: string) => {
    if (!window.confirm('Delete this experiment?')) return
    try {
      await api.deleteExperiment(id)
      toast.success('Experiment deleted')
      refetchExperiments()
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to delete experiment')
    }
  }

  const handleAction = async (id: string, action: 'start' | 'pause' | 'resume' | 'stop') => {
    try {
      await (api as any)[`${action}Experiment`](id)
      toast.success(`Experiment ${action}ed`)
      refetchExperiments()
    } catch (error: any) {
      toast.error(error.response?.data?.detail || `Failed to ${action} experiment`)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-green-500/20 text-green-400'
      case 'created': return 'bg-dark-700 text-dark-400'
      case 'configuring': return 'bg-blue-500/20 text-blue-400'
      case 'paused': return 'bg-yellow-500/20 text-yellow-400'
      case 'completed': return 'bg-purple-500/20 text-purple-400'
      case 'failed': return 'bg-red-500/20 text-red-400'
      case 'stopped': return 'bg-gray-500/20 text-gray-400'
      default: return 'bg-dark-700 text-dark-400'
    }
  }

  return (
    <div className="space-y-6 animate-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-50">Experiments</h1>
          <p className="text-dark-400 mt-1">Federated training experiments</p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-primary-500 text-dark-950 rounded-lg font-medium hover:bg-primary-400 flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          New Experiment
        </button>
      </div>

      <div className="p-6 rounded-xl border border-dark-700 bg-dark-900/50">
        <div className="flex gap-4 mb-6">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-400" />
            <input
              type="text"
              placeholder="Search experiments..."
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
                <th className="pb-3 font-medium text-dark-300">Name</th>
                <th className="pb-3 font-medium text-dark-300">Algorithm</th>
                <th className="pb-3 font-medium text-dark-300">Status</th>
                <th className="pb-3 font-medium text-dark-300">Round</th>
                <th className="pb-3 font-medium text-dark-300">Organisations</th>
                <th className="pb-3 font-medium text-dark-300">Created</th>
                <th className="pb-3 font-medium text-dark-300">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-800">
              {expsLoading ? (
                <tr>
                  <td className="py-4 text-dark-400 text-center" colSpan={7}>
                    <div className="flex items-center justify-center gap-2">
                      <RefreshCw className="w-5 h-5 animate-spin" />
                      Loading...
                    </div>
                  </td>
                </tr>
              ) : experiments && experiments.length > 0 ? (
                (experiments as Experiment[]).map((e) => {
                  const orgNames = organisations
                    ?.filter((o: Organisation) => e.participating_orgs.includes(o.id))
                    .map((o: Organisation) => o.name)
                    .join(', ') || e.participating_orgs.length + ' org(s)'
                  const canStart = e.status === 'created' || e.status === 'configuring'
                  const canPause = e.status === 'running'
                  const canResume = e.status === 'paused'
                  const canStop = e.status === 'running' || e.status === 'paused'
                  return (
                    <tr key={e.id} className="hover:bg-dark-800/50 transition-colors">
                      <td className="py-4 text-dark-50">{e.name}</td>
                      <td className="py-4 text-dark-400 text-capitalize">{e.algorithm}</td>
                      <td className="py-4">
                        <span className={clsx('px-2 py-1 text-xs rounded-full', getStatusColor(e.status))}>
                          {e.status}
                        </span>
                      </td>
                      <td className="py-4 text-dark-300">{e.current_round} / {e.total_rounds}</td>
                      <td className="py-4 text-dark-400 text-sm">{orgNames}</td>
                      <td className="py-4 text-dark-400 text-sm">
                        {e.created_at ? format(new Date(e.created_at), 'MMM d, HH:mm') : '-'}
                      </td>
                      <td className="py-4">
                        <div className="flex items-center gap-2">
                          {canStart && (
                            <button
                              onClick={() => handleAction(e.id, 'start')}
                              className="px-3 py-1.5 text-xs bg-green-500/10 text-green-400 rounded-lg hover:bg-green-500/20 flex items-center gap-1 transition-colors"
                            >
                              <Play className="w-3 h-3" />
                              Start
                            </button>
                          )}
                          {canPause && (
                            <button
                              onClick={() => handleAction(e.id, 'pause')}
                              className="px-3 py-1.5 text-xs bg-yellow-500/10 text-yellow-400 rounded-lg hover:bg-yellow-500/20 flex items-center gap-1 transition-colors"
                            >
                              <Pause className="w-3 h-3" />
                              Pause
                            </button>
                          )}
                          {canResume && (
                            <button
                              onClick={() => handleAction(e.id, 'resume')}
                              className="px-3 py-1.5 text-xs bg-blue-500/10 text-blue-400 rounded-lg hover:bg-blue-500/20 flex items-center gap-1 transition-colors"
                            >
                              <Play className="w-3 h-3" />
                              Resume
                            </button>
                          )}
                          {canStop && (
                            <button
                              onClick={() => handleAction(e.id, 'stop')}
                              className="px-3 py-1.5 text-xs bg-red-500/10 text-red-400 rounded-lg hover:bg-red-500/20 flex items-center gap-1 transition-colors"
                            >
                              <Square className="w-3 h-3" />
                              Stop
                            </button>
                          )}
                          {e.status === 'completed' && (
                            <button
                              onClick={() => {
                                const mId = (e.config as any)?.trained_model_id
                                window.location.href = mId ? `/inference?model_id=${mId}` : '/inference'
                              }}
                              className="px-3 py-1.5 text-xs bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 rounded-lg hover:bg-emerald-500/25 flex items-center gap-1 transition-colors"
                              title="Run Inference with this trained model"
                            >
                              <Zap className="w-3 h-3 text-emerald-400" />
                              Detect
                            </button>
                          )}
                          <button
                            className="px-3 py-1.5 text-xs bg-primary-500/10 text-primary-400 rounded-lg hover:bg-primary-500/20 flex items-center gap-1 transition-colors"
                            onClick={async () => {
                              try {
                                const resp = await api.getExperiment(e.id)
                                setViewExperiment(resp.data)
                              } catch {
                                toast.error('Failed to load experiment details')
                              }
                            }}
                          >
                            <Eye className="w-3 h-3" />
                          </button>
                          <button
                            onClick={() => handleDelete(e.id)}
                            className="px-3 py-1.5 text-xs bg-red-500/10 text-red-400 rounded-lg hover:bg-red-500/20 flex items-center gap-1 transition-colors"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  )
                })
              ) : (
                <tr>
                  <td className="py-4 text-dark-400 text-center" colSpan={7}>No experiments yet</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Create Experiment Modal */}
      <div
        className={clsx('fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm', showCreateModal ? '' : 'hidden')}
      >
        <div className="w-full max-w-2xl bg-dark-900 border border-dark-700 rounded-xl overflow-hidden max-h-[90vh] flex flex-col">
          <div className="p-4 border-b border-dark-700 flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-dark-50">New Model Training Experiment</h2>
              <p className="text-xs text-dark-400">Train Centralized or Federated models on any dataset schema for adaptive intrusion detection</p>
            </div>
            <button
              onClick={() => setShowCreateModal(false)}
              className="p-2 rounded-lg text-dark-400 hover:text-dark-500 hover:bg-dark-800 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          <form onSubmit={handleCreate} className="p-6 space-y-6 overflow-y-auto flex-1">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <label className="text-sm font-medium text-dark-300">Experiment Name</label>
                <input
                  type="text"
                  value={createForm.name}
                  onChange={(e) => setCreateForm({ ...createForm, name: e.target.value })}
                  placeholder="e.g. Kaggle 2025 Intrusion FedAvg"
                  className="w-full px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                  required
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
              <div className="space-y-2 md:col-span-2">
                <label className="text-sm font-medium text-dark-300 flex items-center gap-2">
                  <Database className="w-4 h-4 text-primary-400" />
                  Training Dataset (Any Schema / Format)
                </label>
                <select
                  value={createForm.sample_file ? createForm.sample_file : (createForm.dataset_id ? `ds:${createForm.dataset_id}` : '')}
                  onChange={(e) => {
                    const val = e.target.value
                    if (val.startsWith('ds:')) {
                      setCreateForm({ ...createForm, dataset_id: val.slice(3), sample_file: '' })
                    } else {
                      setCreateForm({ ...createForm, sample_file: val, dataset_id: '' })
                    }
                  }}
                  className="w-full px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 focus:border-primary-500 focus:outline-none"
                >
                  <optgroup label="Latest Kaggle & Benchmark Datasets">
                    <option value="kaggle_cybersecurity_intrusion_2025.csv">Kaggle Cybersecurity Intrusion 2025 (9,537 rows)</option>
                    <option value="kaggle_unsw_nb15_cloud_ids_2025.csv">Kaggle UNSW-NB15 Cloud IDS 2025 (2,000 rows)</option>
                    <option value="kaggle_threat_intel_malicious_ips.csv">Kaggle Threat Intel Malicious IPs (200 rows)</option>
                    <option value="mock_cic_ids_5000_labelled.csv">CIC-IDS2017 Labeled Benchmark (5,000 rows)</option>
                  </optgroup>
                  {registeredDatasets && registeredDatasets.length > 0 && (
                    <optgroup label="Uploaded / Analyzed Datasets">
                      {registeredDatasets.map((d: any) => (
                        <option key={d.id} value={`ds:${d.id}`}>{d.name} ({d.total_rows || d.rows || 0} rows)</option>
                      ))}
                    </optgroup>
                  )}
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-dark-300">Total Rounds</label>
                <input
                  type="number"
                  value={createForm.total_rounds}
                  onChange={(e) => setCreateForm({ ...createForm, total_rounds: parseInt(e.target.value) || 1 })}
                  min="1"
                  max="30"
                  className="w-full px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-dark-300">Target / Label Column (Optional)</label>
                <input
                  type="text"
                  value={createForm.target_column}
                  onChange={(e) => setCreateForm({ ...createForm, target_column: e.target.value })}
                  placeholder="Auto-detect (e.g. Label, attack, is_intrusion)"
                  className="w-full px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 placeholder:text-dark-500 focus:border-primary-500 focus:outline-none"
                />
              </div>
              <div className="space-y-2 md:col-span-2">
                <label className="text-sm font-medium text-dark-300 flex items-center gap-2">
                  <Building2 className="w-4 h-4 text-cyan-400" />
                  Sponsoring / Participating Organisations (1+ for Centralized, 2+ for Federated)
                </label>
                <div className="max-h-36 overflow-y-auto border border-dark-700 rounded-lg p-2 bg-dark-800">
                  {(organisations || []).map((o: Organisation) => (
                    <label key={o.id} className="flex items-center gap-2 p-1.5 rounded hover:bg-dark-700 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={createForm.participating_org_ids.includes(o.id)}
                        onChange={(e) =>
                          setCreateForm({
                            ...createForm,
                            participating_org_ids: e.target.checked
                              ? [...createForm.participating_org_ids, o.id]
                              : createForm.participating_org_ids.filter((id) => id !== o.id),
                          })
                        }
                        className="rounded text-primary-500 focus:ring-primary-500"
                      />
                      <span className="text-dark-200 text-sm font-medium">{o.name}</span>
                    </label>
                  ))}
                  {(organisations?.length ?? 0) === 0 && (
                    <p className="text-xs text-dark-500 text-center">No organisations available. Add one first.</p>
                  )}
                </div>
              </div>
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

            <details className="border border-dark-700 rounded-lg overflow-hidden">
              <summary className="flex items-center justify-between px-4 py-3 bg-dark-800 cursor-pointer text-sm font-medium text-dark-300 hover:bg-dark-700">
                Advanced Config
              </summary>
              <div className="p-4 space-y-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-xs font-medium text-dark-400">Local Epochs</label>
                  <input
                    type="number"
                    value={createForm.config.local_epochs}
                    onChange={(e) =>
                      setCreateForm({ ...createForm, config: { ...createForm.config, local_epochs: parseInt(e.target.value) || 5 } })
                    }
                    className="w-full px-3 py-2 bg-dark-800 border border-dark-700 rounded text-dark-50 focus:border-primary-500 focus:outline-none"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-medium text-dark-400">Batch Size</label>
                  <input
                    type="number"
                    value={createForm.config.batch_size}
                    onChange={(e) =>
                      setCreateForm({ ...createForm, config: { ...createForm.config, batch_size: parseInt(e.target.value) || 128 } })
                    }
                    className="w-full px-3 py-2 bg-dark-800 border border-dark-700 rounded text-dark-50 focus:border-primary-500 focus:outline-none"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-medium text-dark-400">Learning Rate</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={createForm.config.learning_rate}
                    onChange={(e) =>
                      setCreateForm({ ...createForm, config: { ...createForm.config, learning_rate: parseFloat(e.target.value) || 0.001 } })
                    }
                    className="w-full px-3 py-2 bg-dark-800 border border-dark-700 rounded text-dark-50 focus:border-primary-500 focus:outline-none"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-medium text-dark-400">Secure Aggregation</label>
                  <input
                    type="checkbox"
                    checked={createForm.config.secure_aggregation}
                    onChange={(e) =>
                      setCreateForm({ ...createForm, config: { ...createForm.config, secure_aggregation: e.target.checked } })
                    }
                    className="rounded text-primary-500 focus:ring-primary-500"
                  />
                </div>
              </div>
            </details>

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
                {creating ? 'Creating...' : 'Create Experiment'}
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Experiment Details Modal */}
      {viewExperiment && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in">
          <div className="w-full max-w-md bg-dark-900 border border-dark-700 rounded-xl overflow-hidden">
            <div className="p-4 border-b border-dark-700 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-dark-50">Experiment Details</h2>
              <button
                onClick={() => setViewExperiment(null)}
                className="p-2 rounded-lg text-dark-400 hover:text-dark-500 hover:bg-dark-800 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-6 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-dark-500">Name</p>
                  <p className="text-dark-50">{viewExperiment?.name}</p>
                </div>
                <div>
                  <p className="text-xs text-dark-500">Algorithm</p>
                  <p className="text-dark-50">{viewExperiment?.algorithm}</p>
                </div>
                <div>
                  <p className="text-xs text-dark-500">Status</p>
                  <p className="text-dark-50">{viewExperiment?.status}</p>
                </div>
                <div>
                  <p className="text-xs text-dark-500">Round</p>
                  <p className="text-dark-50">{viewExperiment?.current_round} / {viewExperiment?.total_rounds}</p>
                </div>
              </div>
              {viewExperiment?.description && (
                <div>
                  <p className="text-xs text-dark-500">Description</p>
                  <p className="text-dark-300">{viewExperiment?.description}</p>
                </div>
              )}
              <div>
                <p className="text-xs text-dark-500">Organisations</p>
                <p className="text-dark-300 text-sm">{viewExperiment?.participating_orgs.length}</p>
              </div>
              <div>
                <p className="text-xs text-dark-500">Created</p>
                <p className="text-dark-300 text-sm">{viewExperiment?.created_at}</p>
              </div>
            </div>
            <div className="p-4 border-t border-dark-700 flex justify-end">
              <button
                onClick={() => setViewExperiment(null)}
                className="px-4 py-2 bg-primary-500 text-dark-950 rounded-lg font-medium hover:bg-primary-400"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
