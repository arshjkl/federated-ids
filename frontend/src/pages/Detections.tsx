'use client'

import { useState, useCallback, useEffect, useMemo } from 'react'
import { Search, Shield, Download, RefreshCw, Filter, X, ChevronLeft, ChevronRight,
  Eye, Calendar, Database, Brain,
  CheckCircle, FileText, Settings, Zap, Building2 } from 'lucide-react'
import { clsx } from 'clsx'
import { toast } from 'react-hot-toast'
import { useInferenceJobs, useModels, useDatasets, useOrganisations } from '@/hooks/useDashboard'
import { api } from '@/services/api'
import { format, parseISO, startOfDay, endOfDay } from 'date-fns'

interface InferenceJob {
  id: string
  name: string
  model_id: string
  model_name?: string
  model_version?: string
  dataset_id?: string
  status: string
  total_records?: number
  benign_count?: number
  malicious_count?: number
  created_at: string
}

interface DetectionRecord {
  id: string
  inference_job_id: string
  record_index: number
  prediction: number
  probability: number
  confidence: number | null
  true_label: number | null
  features: Record<string, any> | null
  timestamp: string
  job_name: string
  model_name: string | null
  model_version: string | null
  dataset_name: string | null
  organisation_id?: string | null
  organisation_name?: string | null
  predicted_category?: string
  supervised_probability?: number | null
  anomaly_score?: number | null
  rule_score?: number
  risk_score?: number
  severity?: string
  detection_mode?: string
  triggered_rules?: Array<{ rule_id: string; name: string; weight: number; evidence: string }>
  explanation?: string
  top_anomaly_features?: string[]
  compatibility_score?: number
}

interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

interface FilterState {
  search: string
  model_id: string
  dataset_id: string
  organisation_id: string
  prediction: 'all' | 'benign' | 'malicious'
  min_probability: number | null
  max_probability: number | null
  start_date: string
  end_date: string
  date_preset: 'all' | 'today' | 'week' | 'month' | 'custom'
}

const getPredictionColor = (prediction: number) => {
  return prediction === 1 ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'
}

const getPredictionLabel = (prediction: number) => {
  return prediction === 1 ? 'Malicious' : 'Benign'
}

export function Detections() {
  const { data: jobs, refetch: refetchJobs } = useInferenceJobs()
  const { data: registeredModels } = useModels()
  const { data: registeredDatasets } = useDatasets()
  const { data: organisations } = useOrganisations()

  // State
  const [detections, setDetections] = useState<DetectionRecord[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedDetection, setSelectedDetection] = useState<DetectionRecord | null>(null)
  const [showDetails, setShowDetails] = useState(false)
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize, setPageSize] = useState(50)
  const [totalPages, setTotalPages] = useState(0)
  const [totalCount, setTotalCount] = useState(0)
  const [filters, setFilters] = useState<FilterState>({
    search: '',
    model_id: '',
    dataset_id: '',
    organisation_id: '',
    prediction: 'all',
    min_probability: null,
    max_probability: null,
    start_date: '',
    end_date: '',
    date_preset: 'all',
  })
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false)
  const [isExporting, setIsExporting] = useState(false)

  // Derived data
  const completedJobs = useMemo(() => jobs?.filter((j: InferenceJob) => j.status === 'completed') || [], [jobs])
  const uniqueModels = useMemo(() => {
    const models = new Map<string, string>()
    if (Array.isArray(registeredModels)) {
      registeredModels.forEach((m: any) => {
        if (m?.id && m?.name) {
          models.set(m.id, `${m.name} (${m.algorithm})`)
        }
      })
    }
    completedJobs.forEach((job: InferenceJob) => {
      if (job.model_id && !models.has(job.model_id)) {
        models.set(job.model_id, job.model_name || `Model ${job.model_id.slice(0, 8)}...`)
      }
    })
    return Array.from(models.entries()).map(([id, name]) => ({ id, name }))
  }, [completedJobs, registeredModels])
  const uniqueDatasets = useMemo(() => {
    const datasets = new Map<string, string>()
    if (Array.isArray(registeredDatasets)) {
      registeredDatasets.forEach((d: any) => {
        if (d?.id && d?.name) {
          datasets.set(d.id, d.name)
        }
      })
    }
    completedJobs.forEach((job: InferenceJob) => {
      if (job.dataset_id && !datasets.has(job.dataset_id)) {
        datasets.set(job.dataset_id, `Dataset ${job.dataset_id.slice(0, 8)}...`)
      }
    })
    return Array.from(datasets.entries()).map(([id, name]) => ({ id, name }))
  }, [completedJobs, registeredDatasets])

  // Summary stats
  const totalAnalyzed = totalCount || 0
  const maliciousCount = detections.filter(d => d.prediction === 1).length
  const benignCount = detections.filter(d => d.prediction === 0).length
  const threatRate = totalAnalyzed > 0 ? ((maliciousCount / totalAnalyzed) * 100).toFixed(1) : '0'

  // Load detections
  const loadDetections = useCallback(async (page: number = 1) => {
    setIsLoading(true)
    setCurrentPage(page)
    try {
      const params: Record<string, any> = {
        page,
        page_size: pageSize,
      }

      if (filters.search) params.search = filters.search
      if (filters.model_id) params.model_id = filters.model_id
      if (filters.dataset_id) params.dataset_id = filters.dataset_id
      if (filters.organisation_id) params.organisation_id = filters.organisation_id
      if (filters.prediction !== 'all') params.prediction = filters.prediction === 'malicious' ? 1 : 0
      if (filters.min_probability !== null) params.min_probability = filters.min_probability
      if (filters.max_probability !== null) params.max_probability = filters.max_probability
      if (filters.start_date) params.start_date = filters.start_date
      if (filters.end_date) params.end_date = filters.end_date

      const data = await api.getAllDetections(params)
      const response = data.data as PaginatedResponse<DetectionRecord>
      setDetections(response.items)
      setTotalCount(response.total)
      setTotalPages(response.total_pages)
    } catch (error: any) {
      console.error('Failed to load detections:', error)
      toast.error('Failed to load detections')
    } finally {
      setIsLoading(false)
    }
  }, [filters, pageSize])

  // Reload detections whenever filters or pageSize change
  useEffect(() => {
    loadDetections(1)
  }, [loadDetections])

  // Handle date preset changes
  const handleDatePresetChange = (preset: FilterState['date_preset']) => {
    const now = new Date()
    let start = ''
    let end = ''

    switch (preset) {
      case 'today':
        start = format(startOfDay(now), 'yyyy-MM-dd')
        end = format(endOfDay(now), 'yyyy-MM-dd')
        break
      case 'week':
        start = format(startOfDay(new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000)), 'yyyy-MM-dd')
        end = format(endOfDay(now), 'yyyy-MM-dd')
        break
      case 'month':
        start = format(startOfDay(new Date(now.getFullYear(), now.getMonth(), 1)), 'yyyy-MM-dd')
        end = format(endOfDay(now), 'yyyy-MM-dd')
        break
      case 'custom':
      case 'all':
        start = ''
        end = ''
        break
    }

    setFilters(prev => ({
      ...prev,
      date_preset: preset,
      start_date: start,
      end_date: end,
    }))
  }

  // Handle filter changes
  const handleFilterChange = (key: keyof FilterState, value: any) => {
    setFilters(prev => ({ ...prev, [key]: value }))
  }

  const handleSearch = (value: string) => {
    handleFilterChange('search', value)
  }

  const clearFilters = () => {
    setFilters({
      search: '',
      model_id: '',
      dataset_id: '',
      organisation_id: '',
      prediction: 'all',
      min_probability: null,
      max_probability: null,
      start_date: '',
      end_date: '',
      date_preset: 'all',
    })
  }

  const hasActiveFilters = filters.search || filters.model_id || filters.dataset_id ||
    filters.organisation_id || filters.prediction !== 'all' ||
    filters.min_probability !== null || filters.max_probability !== null ||
    filters.start_date || filters.end_date

  // Export CSV
  const handleExport = async () => {
    if (!detections.length) return
    setIsExporting(true)
    try {
      // Use the backend export endpoint for the first job's detections
      // For now, do client-side export with all loaded data
      const headers = [
        'Record ID', 'Job Name', 'Model', 'Dataset', 'Record Index',
        'Prediction', 'Probability', 'Confidence', 'True Label', 'Timestamp'
      ]
      const rows = detections.map(d => [
        d.id,
        d.job_name,
        d.model_name || 'Unknown',
        d.dataset_name || 'Unknown',
        d.record_index,
        getPredictionLabel(d.prediction),
        d.probability.toFixed(6),
        d.confidence?.toFixed(6) || 'N/A',
        d.true_label !== null ? getPredictionLabel(d.true_label) : 'N/A',
        d.timestamp ? format(parseISO(d.timestamp), 'yyyy-MM-dd HH:mm:ss') : 'N/A'
      ])
      const csv = [headers, ...rows].map(r => r.map(v => `"${v}"`).join(',')).join('\n')
      const blob = new Blob([csv], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `detections_${format(new Date(), 'yyyyMMdd_HHmmss')}.csv`
      a.click()
      URL.revokeObjectURL(url)
      toast.success(`Exported ${detections.length} detections`)
    } catch (error) {
      console.error('Export failed:', error)
      toast.error('Export failed')
    } finally {
      setIsExporting(false)
    }
  }

  // View details
  const handleViewDetails = (detection: DetectionRecord) => {
    setSelectedDetection(detection)
    setShowDetails(true)
  }

  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-50">Detections</h1>
          <p className="text-dark-400 mt-1">Individual threat detection records with detailed analysis</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleExport}
            disabled={isExporting || detections.length === 0}
            className="px-4 py-2 bg-green-500/10 text-green-400 rounded-lg hover:bg-green-500/20 flex items-center gap-2 transition-colors text-sm disabled:opacity-50"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </button>
          <button
            onClick={() => void refetchJobs()}
            className="px-4 py-2 bg-primary-500 text-dark-950 rounded-lg font-medium hover:bg-primary-400 flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
        <div className="p-4 rounded-xl border border-dark-700 bg-dark-900/50">
          <p className="text-sm text-dark-400">Total Loaded</p>
          <p className="text-2xl font-bold text-dark-50 mt-1">{detections.length.toLocaleString()}</p>
        </div>
        <div className="p-4 rounded-xl border border-dark-700 bg-dark-900/50">
          <p className="text-sm text-dark-400">Total Records</p>
          <p className="text-2xl font-bold text-dark-50 mt-1">{totalAnalyzed.toLocaleString()}</p>
        </div>
        <div className="p-4 rounded-xl border border-dark-700 bg-dark-900/50">
          <p className="text-sm text-dark-400">Benign</p>
          <p className="text-2xl font-bold text-green-400 mt-1">{benignCount.toLocaleString()}</p>
        </div>
        <div className="p-4 rounded-xl border border-dark-700 bg-dark-900/50">
          <p className="text-sm text-dark-400">Malicious</p>
          <p className="text-2xl font-bold text-red-400 mt-1">{maliciousCount.toLocaleString()}</p>
        </div>
        <div className="p-4 rounded-xl border border-dark-700 bg-dark-900/50">
          <p className="text-sm text-dark-400">Threat Rate</p>
          <p className="text-2xl font-bold text-dark-50 mt-1">{threatRate}%</p>
        </div>
      </div>

      {/* Filters Panel */}
      <div className="p-6 rounded-xl border border-dark-700 bg-dark-900/50">
        {/* Basic Filters */}
        <div className="flex flex-wrap gap-4 mb-4">
          <div className="relative flex-1 min-w-[250px] max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-400" />
            <input
              type="text"
              placeholder="Search detections (job name, model, dataset)..."
              value={filters.search}
              onChange={(e) => handleSearch(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 placeholder-dark-400 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
            />
          </div>

          <select
            value={filters.prediction}
            onChange={(e) => handleFilterChange('prediction', e.target.value)}
            className="px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500 min-w-[150px]"
          >
            <option value="all">All Predictions</option>
            <option value="benign">Benign Only</option>
            <option value="malicious">Malicious Only</option>
          </select>

          <select
            value={filters.model_id}
            onChange={(e) => handleFilterChange('model_id', e.target.value)}
            className="px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500 min-w-[180px]"
          >
            <option value="">All Models</option>
            {uniqueModels.map((m) => (
              <option key={m.id} value={m.id}>{m.name}</option>
            ))}
          </select>

          <select
            value={filters.dataset_id}
            onChange={(e) => handleFilterChange('dataset_id', e.target.value)}
            className="px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500 min-w-[180px]"
          >
            <option value="">All Datasets</option>
            {uniqueDatasets.map((d) => (
              <option key={d.id} value={d.id}>{d.name}</option>
            ))}
          </select>

          <select
            value={filters.organisation_id}
            onChange={(e) => handleFilterChange('organisation_id', e.target.value)}
            className="px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-cyan-300 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500 min-w-[180px]"
          >
            <option value="">All Organisations</option>
            {(organisations || []).map((o: any) => (
              <option key={o.id} value={o.id}>{o.name}</option>
            ))}
          </select>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}
              className={clsx(
                'px-4 py-2 rounded-lg flex items-center gap-2 transition-colors',
                showAdvancedFilters
                  ? 'bg-primary-500/10 text-primary-400 border border-primary-500/30'
                  : 'bg-dark-800 text-dark-300 border border-dark-700 hover:bg-dark-700'
              )}
            >
              <Filter className="w-4 h-4" />
              Advanced
              {hasActiveFilters && (
                <span className="px-1.5 py-0.5 text-xs bg-red-500/20 text-red-400 rounded-full">
                  {[
                    filters.model_id, filters.dataset_id, filters.prediction !== 'all',
                    filters.min_probability !== null, filters.max_probability !== null,
                    filters.start_date, filters.end_date
                  ].filter(Boolean).length}
                </span>
              )}
            </button>
          </div>
        </div>

        {/* Advanced Filters */}
        {showAdvancedFilters && (
          <div className="border-t border-dark-700 pt-4 animate-in">
            <div className="flex flex-wrap gap-4 mb-4">
              {/* Date Preset */}
              <div className="flex items-center gap-2">
                <Calendar className="w-4 h-4 text-dark-400" />
                <span className="text-sm text-dark-400">Date Range:</span>
                <select
                  value={filters.date_preset}
                  onChange={(e) => handleDatePresetChange(e.target.value as FilterState['date_preset'])}
                  className="px-3 py-1.5 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 text-sm focus:border-primary-500"
                >
                  <option value="all">All Time</option>
                  <option value="today">Today</option>
                  <option value="week">Last 7 Days</option>
                  <option value="month">This Month</option>
                  <option value="custom">Custom Range</option>
                </select>
              </div>

              {/* Custom Date Range */}
              {filters.date_preset === 'custom' && (
                <>
                  <div className="flex items-center gap-1">
                    <input
                      type="date"
                      value={filters.start_date}
                      onChange={(e) => handleFilterChange('start_date', e.target.value)}
                      className="px-3 py-1.5 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 text-sm focus:border-primary-500"
                    />
                    <span className="text-dark-500">to</span>
                    <input
                      type="date"
                      value={filters.end_date}
                      onChange={(e) => handleFilterChange('end_date', e.target.value)}
                      className="px-3 py-1.5 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 text-sm focus:border-primary-500"
                    />
                  </div>
                </>
              )}

              {/* Probability Range */}
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4 text-dark-400" />
                <span className="text-sm text-dark-400">Probability:</span>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  placeholder="Min"
                  value={filters.min_probability ?? ''}
                  onChange={(e) => handleFilterChange('min_probability', e.target.value ? parseFloat(e.target.value) : null)}
                  className="w-24 px-2 py-1.5 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 text-sm focus:border-primary-500"
                />
                <span className="text-dark-500">-</span>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  placeholder="Max"
                  value={filters.max_probability ?? ''}
                  onChange={(e) => handleFilterChange('max_probability', e.target.value ? parseFloat(e.target.value) : null)}
                  className="w-24 px-2 py-1.5 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 text-sm focus:border-primary-500"
                />
              </div>
            </div>

            {/* Clear Filters */}
            {hasActiveFilters && (
              <button
                onClick={clearFilters}
                className="px-3 py-1.5 text-sm bg-red-500/10 text-red-400 rounded-lg hover:bg-red-500/20 flex items-center gap-1 transition-colors"
              >
                <X className="w-4 h-4" />
                Clear All Filters
              </button>
            )}
          </div>
        )}
      </div>

      {/* Detections Table */}
      <div className="p-6 rounded-xl border border-dark-700 bg-dark-900/50">
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-dark-700">
                <th className="pb-3 font-medium text-dark-300 w-12">#</th>
                <th className="pb-3 font-medium text-dark-300">Timestamp</th>
                <th className="pb-3 font-medium text-dark-300">Organisation</th>
                <th className="pb-3 font-medium text-dark-300">Job</th>
                <th className="pb-3 font-medium text-dark-300">Model</th>
                <th className="pb-3 font-medium text-dark-300">Mode</th>
                <th className="pb-3 font-medium text-dark-300">Dataset</th>
                <th className="pb-3 font-medium text-dark-300">Predicted Category</th>
                <th className="pb-3 font-medium text-dark-300">Severity</th>
                <th className="pb-3 font-medium text-dark-300">Risk Score</th>
                <th className="pb-3 font-medium text-dark-300">Confidence</th>
                <th className="pb-3 font-medium text-dark-300 w-12">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-800">
              {isLoading ? (
                <tr>
                  <td className="py-4 text-dark-400 text-center" colSpan={12}>
                    <div className="flex items-center justify-center gap-2">
                      <RefreshCw className="w-5 h-5 animate-spin" />
                      Loading detections...
                    </div>
                  </td>
                </tr>
              ) : detections.length > 0 ? (
                detections.map((detection) => (
                  <tr key={detection.id} className="hover:bg-dark-800/50 transition-colors cursor-pointer"
                    onClick={() => handleViewDetails(detection)}
                  >
                    <td className="py-3 text-dark-400 text-sm font-mono">{detection.record_index}</td>
                    <td className="py-3 text-dark-400 text-sm font-mono">
                      {detection.timestamp ? format(parseISO(detection.timestamp), 'MMM d, HH:mm:ss') : '-'}
                    </td>
                    <td className="py-3 text-cyan-300 text-xs truncate max-w-[130px]" title={detection.organisation_name || 'Cyber Defense Consortium'}>
                      <div className="flex items-center gap-1.5">
                        <Building2 className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                        <span className="truncate">{detection.organisation_name || 'Cyber Defense Consortium'}</span>
                      </div>
                    </td>
                    <td className="py-3 text-dark-50 truncate max-w-[140px]" title={detection.job_name}>
                      {detection.job_name}
                    </td>
                    <td className="py-3 text-emerald-300 font-mono text-xs truncate max-w-[170px]" title={detection.model_name || 'Unknown'}>
                      {detection.model_name || 'Unknown'}
                    </td>
                    <td className="py-3 text-primary-400 font-mono text-xs uppercase">
                      {detection.detection_mode || 'supervised'}
                    </td>
                    <td className="py-3 text-dark-400 font-mono text-xs truncate max-w-[120px]" title={detection.dataset_name || ''}>
                      {detection.dataset_name || 'Unknown'}
                    </td>
                    <td className="py-3">
                      <span className={clsx('px-2 py-1 text-xs rounded-full font-medium', getPredictionColor(detection.prediction))}>
                        {detection.predicted_category || getPredictionLabel(detection.prediction)}
                      </span>
                    </td>
                    <td className="py-3">
                      <span className="px-2 py-0.5 text-[11px] rounded font-mono uppercase bg-dark-800 text-dark-200 border border-dark-700">
                        {detection.severity || 'info'}
                      </span>
                    </td>
                    <td className="py-3 text-dark-300 font-mono">
                      {(detection.risk_score ?? detection.probability).toFixed(4)}
                      <div className="w-20 h-1.5 bg-dark-700 rounded-full mt-1 overflow-hidden">
                        <div
                          className={clsx('h-full rounded-full transition-all',
                            detection.prediction === 1 ? 'bg-red-500' : 'bg-green-500'
                          )}
                          style={{ width: `${((detection.risk_score ?? detection.probability) * 100).toFixed(1)}%` }}
                        />
                      </div>
                    </td>
                    <td className="py-3 text-dark-300 font-mono">
                      {detection.confidence !== null ? detection.confidence.toFixed(4) : 'N/A'}
                    </td>
                    <td className="py-3 text-center">
                      <button
                        onClick={(e) => { e.stopPropagation(); handleViewDetails(detection); }}
                        className="p-1.5 text-dark-400 hover:text-primary-400 hover:bg-dark-800 rounded transition-colors"
                        title="View Details"
                      >
                        <Eye className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td className="py-12 text-dark-400 text-center" colSpan={10}>
                    <div className="flex flex-col items-center gap-3">
                      <Shield className="w-12 h-12 text-dark-600" />
                      <p>No detections found</p>
                      <p className="text-sm">Adjust filters or run inference to see results</p>
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="mt-4 flex items-center justify-between">
            <div className="text-sm text-dark-400">
              Page {currentPage} of {totalPages} • {totalCount.toLocaleString()} total records
            </div>
            <div className="flex items-center gap-2">
              <select
                value={pageSize}
                onChange={(e) => { setPageSize(Number(e.target.value)); loadDetections(1); }}
                className="px-3 py-1.5 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 text-sm focus:border-primary-500"
              >
                <option value={25}>25 per page</option>
                <option value={50}>50 per page</option>
                <option value={100}>100 per page</option>
                <option value={200}>200 per page</option>
              </select>
              <button
                onClick={() => loadDetections(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1.5 bg-dark-800 border border-dark-700 rounded-lg text-dark-300 hover:bg-dark-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="text-dark-400 px-4">
                Page {currentPage} of {totalPages}
              </span>
              <button
                onClick={() => loadDetections(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-1.5 bg-dark-800 border border-dark-700 rounded-lg text-dark-300 hover:bg-dark-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Details Panel */}
      {showDetails && selectedDetection && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in">
          <div className="w-full max-w-4xl max-h-[90vh] bg-dark-900 border border-dark-700 rounded-xl overflow-hidden flex flex-col">
            <div className="p-4 border-b border-dark-700 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={clsx('p-2 rounded-lg', getPredictionColor(selectedDetection.prediction))}>
                  <Shield className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-dark-50">Detection Details</h2>
                  <p className="text-sm text-dark-400">Record #{selectedDetection.record_index} • {selectedDetection.job_name}</p>
                </div>
              </div>
              <button onClick={() => { setShowDetails(false); setSelectedDetection(null); }} className="p-2 rounded-lg text-dark-400 hover:text-dark-500 hover:bg-dark-800 transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex-1 overflow-auto p-4">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Main Info */}
                <div className="lg:col-span-2 space-y-4">
                  {/* Prediction Summary */}
                  <div className={clsx('p-4 rounded-lg border', getPredictionColor(selectedDetection.prediction).replace('bg-', 'border-').replace('text-', 'bg-').replace('/20', '/10'))}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className={clsx('p-3 rounded-full', getPredictionColor(selectedDetection.prediction))}>
                          <span className={clsx('text-2xl font-bold', selectedDetection.prediction === 1 ? 'text-red-400' : 'text-green-400')}>
                            {selectedDetection.prediction === 1 ? '⚠' : '✓'}
                          </span>
                        </div>
                        <div>
                          <h3 className="text-lg font-semibold text-dark-50">
                            {selectedDetection.predicted_category || `${getPredictionLabel(selectedDetection.prediction)} Traffic Detected`}
                          </h3>
                          <p className="text-sm text-dark-400">
                            Risk Score: {((selectedDetection.risk_score ?? selectedDetection.probability) * 100).toFixed(1)}% •
                            Severity: {(selectedDetection.severity || 'informational').toUpperCase()} •
                            Mode: {(selectedDetection.detection_mode || 'supervised').toUpperCase()}
                          </p>
                        </div>
                      </div>
                      {selectedDetection.true_label !== null && (
                        <div className="text-right">
                          <p className="text-xs text-dark-400">True Label</p>
                          <span className={clsx('px-2 py-1 text-sm rounded-full font-medium', getPredictionColor(selectedDetection.true_label))}>
                            {getPredictionLabel(selectedDetection.true_label)}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Explanation & Triggered Rules */}
                  {selectedDetection.explanation && (
                    <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg space-y-2">
                      <h4 className="text-sm font-semibold text-dark-50">Adaptive Detection Explanation</h4>
                      <p className="text-xs text-dark-300 leading-relaxed">{selectedDetection.explanation}</p>
                      <div className="grid grid-cols-3 gap-2 pt-2 text-xs font-mono">
                        <div className="p-2 bg-dark-900 rounded">
                          <span className="text-dark-400 block">Supervised Prob</span>
                          <span className="text-primary-400">{selectedDetection.supervised_probability !== null && selectedDetection.supervised_probability !== undefined ? selectedDetection.supervised_probability.toFixed(4) : 'N/A'}</span>
                        </div>
                        <div className="p-2 bg-dark-900 rounded">
                          <span className="text-dark-400 block">Anomaly Score</span>
                          <span className="text-yellow-400">{selectedDetection.anomaly_score !== null && selectedDetection.anomaly_score !== undefined ? selectedDetection.anomaly_score.toFixed(4) : 'N/A'}</span>
                        </div>
                        <div className="p-2 bg-dark-900 rounded">
                          <span className="text-dark-400 block">Rule Score</span>
                          <span className="text-blue-400">{(selectedDetection.rule_score ?? 0).toFixed(4)}</span>
                        </div>
                      </div>
                      {selectedDetection.triggered_rules && selectedDetection.triggered_rules.length > 0 && (
                        <div className="pt-2 space-y-1">
                          <p className="text-xs font-semibold text-orange-400">Triggered Network Heuristic Rules:</p>
                          {selectedDetection.triggered_rules.map((r, idx) => (
                            <div key={idx} className="text-xs font-mono text-dark-300 bg-dark-900 p-1.5 rounded flex justify-between">
                              <span>{r.name}</span>
                              <span className="text-dark-400">{r.evidence}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Metadata */}
                  <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
                    <h4 className="text-sm font-semibold text-dark-50 mb-3 flex items-center gap-2">
                      <Database className="w-4 h-4 text-primary-400" />
                      Inference Job Details
                    </h4>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-dark-400">Job ID</p>
                        <p className="font-mono text-dark-300">{selectedDetection.inference_job_id}</p>
                      </div>
                      <div>
                        <p className="text-dark-400">Job Name</p>
                        <p className="text-dark-300 truncate">{selectedDetection.job_name}</p>
                      </div>
                      <div>
                        <p className="text-dark-400">Model</p>
                        <p className="text-dark-300 font-mono">{selectedDetection.model_name || 'Unknown'}</p>
                      </div>
                      <div>
                        <p className="text-dark-400">Model Version</p>
                        <p className="text-dark-300 font-mono text-xs">{selectedDetection.model_version || 'N/A'}</p>
                      </div>
                      <div>
                        <p className="text-dark-400">Organisation</p>
                        <p className="text-cyan-300 font-medium truncate flex items-center gap-1.5">
                          <Building2 className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                          {selectedDetection.organisation_name || 'Cyber Defense Consortium'}
                        </p>
                      </div>
                      <div>
                        <p className="text-dark-400">Dataset</p>
                        <p className="text-dark-300 truncate">{selectedDetection.dataset_name || 'Unknown'}</p>
                      </div>
                      <div>
                        <p className="text-dark-400">Record Index</p>
                        <p className="font-mono text-dark-300">{selectedDetection.record_index}</p>
                      </div>
                      <div>
                        <p className="text-dark-400">Timestamp</p>
                        <p className="text-dark-300">
                          {selectedDetection.timestamp ? format(parseISO(selectedDetection.timestamp), 'yyyy-MM-dd HH:mm:ss') : 'N/A'}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Features */}
                  {selectedDetection.features && Object.keys(selectedDetection.features).length > 0 && (
                    <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="text-sm font-semibold text-dark-50 flex items-center gap-2">
                          <Settings className="w-4 h-4 text-primary-400" />
                          Input Features ({Object.keys(selectedDetection.features).length})
                        </h4>
                        <button
                          onClick={() => {
                            const blob = new Blob([JSON.stringify(selectedDetection.features, null, 2)], { type: 'application/json' })
                            const url = URL.createObjectURL(blob)
                            const a = document.createElement('a')
                            a.href = url
                            a.download = `features_${selectedDetection.id}.json`
                            a.click()
                            URL.revokeObjectURL(url)
                          }}
                          className="px-2 py-1 text-xs bg-dark-700 text-dark-300 rounded hover:bg-dark-600"
                        >
                          <Download className="w-3 h-3 mr-1" />
                          Export JSON
                        </button>
                      </div>
                      <div className="overflow-x-auto max-h-96">
                        <table className="w-full text-xs min-w-max">
                          <thead className="sticky top-0 bg-dark-900 border-b border-dark-700 z-10">
                            <tr>
                              <th className="pb-2 font-medium text-dark-300 text-left px-2">Feature</th>
                              <th className="pb-2 font-medium text-dark-300 text-left px-2">Value</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-dark-800">
                            {Object.entries(selectedDetection.features).map(([key, value]) => (
                              <tr key={key} className="hover:bg-dark-800/50">
                                <td className="py-1.5 text-dark-400 font-mono text-xs px-2 max-w-[200px] truncate" title={key}>{key}</td>
                                <td className="py-1.5 text-dark-300 font-mono px-2">
                                  {typeof value === 'number' ? value.toFixed(4) : value}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>

                {/* Sidebar */}
                <div className="space-y-4">
                  {/* Probability Gauge */}
                  <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
                    <h4 className="text-sm font-semibold text-dark-50 mb-3 flex items-center gap-2">
                      <Zap className="w-4 h-4 text-primary-400" />
                      Attack Probability
                    </h4>
                    <div className="text-center">
                      <div className="relative w-48 h-48 mx-auto mb-4">
                        <svg viewBox="0 0 100 100">
                          <circle
                            cx="50" cy="50" r="40"
                            stroke="#1e293b" strokeWidth="8" fill="none"
                          />
                          <circle
                            cx="50" cy="50" r="40"
                            stroke={selectedDetection.prediction === 1 ? '#ef4444' : '#22c55e'}
                            strokeWidth="8" fill="none"
                            strokeDasharray={`${selectedDetection.probability * 100} 100`}
                            strokeLinecap="round"
                            transform="rotate(-90 50 50)"
                            style={{ transition: 'stroke-dasharray 0.5s' }}
                          />
                        </svg>
                        <div className="absolute inset-0 flex items-center justify-center">
                          <span className="text-3xl font-bold text-dark-50">
                            {(selectedDetection.probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                      <p className="text-xs text-dark-400">
                        Threshold: {(selectedDetection.prediction === 1 ? '≥' : '<')} 50%
                      </p>
                    </div>
                  </div>

                  {/* Confidence */}
                  <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
                    <h4 className="text-sm font-semibold text-dark-50 mb-3 flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-primary-400" />
                      Confidence
                    </h4>
                    <div className="text-center">
                      <p className="text-3xl font-bold text-dark-50">
                        {selectedDetection.confidence !== null ? (selectedDetection.confidence * 100).toFixed(1) + '%' : 'N/A'}
                      </p>
                      <p className="text-xs text-dark-400 mt-1">Model certainty in prediction</p>
                    </div>
                  </div>

                  {/* Quick Stats */}
                  <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
                    <h4 className="text-sm font-semibold text-dark-50 mb-3 flex items-center gap-2">
                      <FileText className="w-4 h-4 text-primary-400" />
                      Quick Stats
                    </h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-dark-400">Prediction</span>
                        <span className={clsx('font-medium', getPredictionColor(selectedDetection.prediction))}>
                          {getPredictionLabel(selectedDetection.prediction)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-dark-400">Probability</span>
                        <span className="font-mono text-dark-300">{selectedDetection.probability.toFixed(4)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-dark-400">Confidence</span>
                        <span className="font-mono text-dark-300">{selectedDetection.confidence?.toFixed(4) || 'N/A'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-dark-400">True Label</span>
                        <span className={selectedDetection.true_label !== null ? clsx('font-medium', getPredictionColor(selectedDetection.true_label)) : 'text-dark-500'}>
                          {selectedDetection.true_label !== null ? getPredictionLabel(selectedDetection.true_label) : 'N/A'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}