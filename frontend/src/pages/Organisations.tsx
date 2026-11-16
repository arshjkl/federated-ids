'use client'

import { useState } from 'react'
import { Plus, Search, Filter, Trash2, Edit2, Eye } from 'lucide-react'
import { clsx } from 'clsx'
import { toast } from 'react-hot-toast'
import { useOrganisations } from '@/hooks/useDashboard'
import { api } from '@/services/api'

interface Organisation {
  id: string
  name: string
  org_id: string
  description?: string
  status: string
  dataset?: string
  training_rounds_participated: number
  total_samples: number
  created_at: string
}

export function Organisations() {
  const { data: organisations, isLoading, refetch } = useOrganisations()

  const [showCreateModal, setShowCreateModal] = useState(false)
  const [creating, setCreating] = useState(false)
  const [viewOrg, setViewOrg] = useState<Organisation | null>(null)
  const [editOrg, setEditOrg] = useState<Organisation | null>(null)
  const [createForm, setCreateForm] = useState({
    name: '',
    org_id: '',
    description: '',
    dataset_id: '',
  })

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault()
    setCreating(true)
    try {
      const payload: any = {
        name: createForm.name,
        org_id: createForm.org_id,
        description: createForm.description,
      }
      if (createForm.dataset_id) {
        payload.dataset_id = createForm.dataset_id
      }
      if (editOrg) {
        await api.updateOrganisation(editOrg.id, payload)
        toast.success('Organisation updated')
      } else {
        await api.createOrganisation(payload)
        toast.success('Organisation added')
      }
      setShowCreateModal(false)
      setEditOrg(null)
      setCreateForm({ name: '', org_id: '', description: '', dataset_id: '' })
      refetch()
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to save organisation')
    } finally {
      setCreating(false)
    }
  }

  const handleDelete = async (id: string) => {
    if (!window.confirm('Delete this organisation?')) return
    try {
      await api.deleteOrganisation(id)
      toast.success('Organisation deleted')
      refetch()
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to delete organisation')
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500/20 text-green-400'
      case 'training': return 'bg-blue-500/20 text-blue-400'
      case 'paused': return 'bg-yellow-500/20 text-yellow-400'
      case 'inactive': return 'bg-gray-500/20 text-gray-400'
      default: return 'bg-dark-700 text-dark-400'
    }
  }

  return (
    <div className="space-y-6 animate-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-50">Organisations</h1>
          <p className="text-dark-400 mt-1">Manage participating organisations</p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-primary-500 text-dark-950 rounded-lg font-medium hover:bg-primary-400 flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          Add Organisation
        </button>
      </div>

      <div className="p-6 rounded-xl border border-dark-700 bg-dark-900/50">
        <div className="flex gap-4 mb-6">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-400" />
            <input
              type="text"
              placeholder="Search organisations..."
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
                <th className="pb-3 font-medium text-dark-300">Org ID</th>
                <th className="pb-3 font-medium text-dark-300">Status</th>
                <th className="pb-3 font-medium text-dark-300">Rounds</th>
                <th className="pb-3 font-medium text-dark-300">Samples</th>
                <th className="pb-3 font-medium text-dark-300">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-800">
              {isLoading ? (
                <tr>
                  <td className="py-4 text-dark-400 text-center" colSpan={6}>
                    <div className="flex items-center justify-center gap-2">
                      <Eye className="w-5 h-5 animate-spin" />
                      Loading...
                    </div>
                  </td>
                </tr>
              ) : organisations && organisations.length > 0 ? (
                (organisations as Organisation[]).map((o) => (
                  <tr key={o.id} className="hover:bg-dark-800/50 transition-colors">
                    <td className="py-4 text-dark-50">{o.name}</td>
                    <td className="py-4 text-dark-400 font-mono text-sm">{o.org_id}</td>
                    <td className="py-4">
                      <span className={clsx('px-2 py-1 text-xs rounded-full', getStatusColor(o.status))}>
                        {o.status}
                      </span>
                    </td>
                    <td className="py-4 text-dark-300">{o.training_rounds_participated}</td>
                    <td className="py-4 text-dark-300">{o.total_samples.toLocaleString()}</td>
                    <td className="py-4">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => setViewOrg(o)}
                          className="px-3 py-1.5 text-sm bg-dark-700 text-dark-300 rounded-lg hover:bg-dark-600 flex items-center gap-1 transition-colors"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => {
                            setEditOrg(o);
                            setCreateForm({
                              name: o.name,
                              org_id: o.org_id,
                              description: o.description || '',
                              dataset_id: o.dataset || '',
                            });
                            setShowCreateModal(true);
                          }}
                          className="px-3 py-1.5 text-sm bg-primary-500/10 text-primary-400 rounded-lg hover:bg-primary-500/20 flex items-center gap-1 transition-colors"
                        >
                          <Edit2 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleDelete(o.id)}
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
                  <td className="py-4 text-dark-400 text-center" colSpan={6}>No organisations yet</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Create Organisation Modal */}
      <div
        className={clsx('fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm', showCreateModal ? '' : 'hidden')}
      >
        <div className="w-full max-w-md bg-dark-900 border border-dark-700 rounded-xl overflow-hidden">
          <div className="p-4 border-b border-dark-700 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-dark-50">Add Organisation</h2>
            <button
              onClick={() => setShowCreateModal(false)}
              className="p-2 rounded-lg text-dark-400 hover:text-dark-500 hover:bg-dark-800 transition-colors"
            >
              <Eye className="w-5 h-5" />
            </button>
          </div>
          <form onSubmit={handleCreate} className="p-6 space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-dark-300">Name</label>
              <input
                type="text"
                value={createForm.name}
                onChange={(e) => setCreateForm({ ...createForm, name: e.target.value })}
                placeholder="Organisation name"
                className="w-full px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                required
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-dark-300">Org ID</label>
              <input
                type="text"
                value={createForm.org_id}
                onChange={(e) => setCreateForm({ ...createForm, org_id: e.target.value })}
                placeholder="e.g. org-001"
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
              <label className="text-sm font-medium text-dark-300">Dataset ID (optional)</label>
              <input
                type="text"
                value={createForm.dataset_id}
                onChange={(e) => setCreateForm({ ...createForm, dataset_id: e.target.value })}
                placeholder="Dataset UUID"
                className="w-full px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
              />
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
                {creating ? 'Adding...' : 'Add Organisation'}
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* View Organisation Modal */}
      <div
        className={clsx(
          'fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm',
          viewOrg ? '' : 'hidden'
        )}
      >
        <div className="w-full max-w-md bg-dark-900 border border-dark-700 rounded-xl overflow-hidden">
          <div className="p-4 border-b border-dark-700 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-dark-50">Organisation Details</h2>
            <button
              onClick={() => setViewOrg(null)}
              className="p-2 rounded-lg text-dark-400 hover:text-dark-500 hover:bg-dark-800 transition-colors"
            >
              <Eye className="w-5 h-5" />
            </button>
          </div>
          <div className="p-6 space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-dark-300">Name</label>
              <p className="text-dark-50">{viewOrg?.name}</p>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-dark-300">Org ID</label>
              <p className="text-dark-400 font-mono text-sm">{viewOrg?.org_id}</p>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-dark-300">Description</label>
              <p className="text-dark-400">{viewOrg?.description || '—'}</p>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-dark-300">Status</label>
              <span
                className={clsx(
                  'px-2 py-1 text-xs rounded-full',
                  getStatusColor(viewOrg?.status || 'inactive')
                )}
              >
                {viewOrg?.status}
              </span>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-dark-300">Rounds Participated</label>
              <p className="text-dark-300">{viewOrg?.training_rounds_participated}</p>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-dark-300">Total Samples</label>
              <p className="text-dark-300">{viewOrg?.total_samples?.toLocaleString() || '0'}</p>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-dark-300">Dataset ID</label>
              <p className="text-dark-400 font-mono text-sm">{viewOrg?.dataset || '—'}</p>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-dark-300">Created At</label>
              <p className="text-dark-400 font-mono text-sm">{viewOrg?.created_at}</p>
            </div>
          </div>
          <div className="p-6 border-t border-dark-700 flex justify-end gap-3">
            <button
              onClick={() => setViewOrg(null)}
              className="px-4 py-2 bg-dark-800 text-dark-300 rounded-lg hover:bg-dark-700"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
