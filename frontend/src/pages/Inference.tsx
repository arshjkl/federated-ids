'use client'

import { useState } from 'react'
import {
  Plus, Search, Filter, Play, Upload, FileText, Download, Trash2,
  Loader2, X, ChevronLeft, ChevronRight
} from 'lucide-react'
import { clsx } from 'clsx'
import { useInferenceJobs } from '@/hooks/useDashboard'
import { api } from '@/services/api'
import { format } from 'date-fns'

interface DetectionResult {
  id: string
  record_index: number
  prediction: number
  probability: number
  confidence: number
  true_label?: number
  features?: Record<string, number>
  timestamp: string
}

interface InferenceJob {
  id: string
  name: string
  model_id: string
  dataset_id: string
  organisation_id?: string
  status: string
  config?: { threshold: number; batch_size: number }
  total_records?: number
  benign_count?: number
  malicious_count?: number
  processing_time?: number
  results_path?: string
  error_message?: string
  started_at?: string
  completed_at?: string
  created_at: string
}

function ResultsModal({
  isOpen,
  onClose,
  job,
  results,
  isLoading,
  filteredResults,
  resultsPage,
  setResultsPage,
  filterPrediction,
  setFilterPrediction,
  exportResults,
}: {
  isOpen: boolean
  onClose: () => void
  job: InferenceJob | null
  results: DetectionResult[]
  isLoading: boolean
  filteredResults: DetectionResult[]
  resultsPage: number
  setResultsPage: (page: number) => void
  filterPrediction: 'all' | 'benign' | 'malicious'
  setFilterPrediction: (v: 'all' | 'benign' | 'malicious') => void
  exportResults: () => void
}) {
  if (!isOpen || !job) return null

  const benignCount = results.filter(r => r.prediction === 0).length
  const maliciousCount = results.filter(r => r.prediction === 1).length
  const totalPages = Math.ceil(filteredResults.length / 20)
  const paginatedResults = filteredResults.slice(
    (resultsPage - 1) * 20,
    resultsPage * 20
  )

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in">
      <div className="w-full max-w-6xl max-h-[90vh] bg-dark-900 border border-dark-700 rounded-xl overflow-hidden flex flex-col">
        <div className="p-4 border-b border-dark-700 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary-500/10 text-primary-400">
              <FileText className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-dark-50">{job.name}</h2>
              <p className="text-sm text-dark-400">Inference Results</p>
            </div>
          </div>
          <button onClick={onClose} className="p-2 rounded-lg text-dark-400 hover:text-dark-500 hover:bg-dark-800 transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-4 border-b border-dark-700 bg-dark-800/50 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-dark-50">{results.length.toLocaleString()}</p>
            <p className="text-xs text-dark-400">Total Records</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-green-400">{benignCount.toLocaleString()}</p>
            <p className="text-xs text-dark-400">Benign</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-red-400">{maliciousCount.toLocaleString()}</p>
            <p className="text-xs text-dark-400">Malicious</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-primary-400">{(maliciousCount / Math.max(results.length, 1) * 100).toFixed(1)}%</p>
            <p className="text-xs text-dark-400">Malicious Rate</p>
          </div>
        </div>

        <div className="p-4 border-b border-dark-700 flex flex-wrap items-center gap-4">
          <div className="flex gap-2">
            {(['all', 'benign', 'malicious'] as const).map((filter) => (
              <button
                key={filter}
                onClick={() => setFilterPrediction(filter)}
                className={clsx(
                  'px-3 py-1.5 text-sm rounded-lg transition-colors',
                  filterPrediction === filter
                    ? 'bg-primary-500 text-dark-950'
                    : 'bg-dark-800 text-dark-300 hover:bg-dark-700'
                )}
              >
                {filter.charAt(0).toUpperCase() + filter.slice(1)}
                <span className={clsx('ml-1.5 px-1.5 py-0.5 text-xs rounded',
                  filter === 'all' ? 'bg-dark-700' :
                  filter === 'benign' ? 'bg-green-500/20 text-green-400' :
                  'bg-red-500/20 text-red-400'
                )}>
                  {filteredResults.filter(r =>
                    filter === 'all' ? true : filter === 'benign' ? r.prediction === 0 : r.prediction === 1
                  ).length}
                </span>
              </button>
            ))}
          </div>
          <div className="ml-auto flex items-center gap-2">
            <button onClick={exportResults} className="px-4 py-2 bg-green-500/10 text-green-400 rounded-lg hover:bg-green-500/20 flex items-center gap-2 transition-colors text-sm">
              <Download className="w-4 h-4" />
              Export CSV
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-auto p-4">
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <Loader2 className="w-8 h-8 animate-spin text-primary-400" />
            </div>
          ) : filteredResults.length === 0 ? (
            <div className="flex items-center justify-center h-64 text-dark-400">
              No results matching filter
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-dark-900 border-b border-dark-700 z-10">
                  <tr>
                    <th className="pb-2 font-medium text-dark-300 text-left">Index</th>
                    <th className="pb-2 font-medium text-dark-300 text-left">Prediction</th>
                    <th className="pb-2 font-medium text-dark-300 text-left">Probability</th>
                    <th className="pb-2 font-medium text-dark-300 text-left">Confidence</th>
                    <th className="pb-2 font-medium text-dark-300 text-left">True Label</th>
                    <th className="pb-2 font-medium text-dark-300 text-left">Timestamp</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-dark-800">
                  {paginatedResults.map((result) => (
                    <tr key={result.id} className="hover:bg-dark-800/50">
                      <td className="py-2 text-dark-400 font-mono">{result.record_index}</td>
                      <td className="py-2">
                        <span className={clsx(
                          'px-2 py-0.5 text-xs rounded-full font-medium',
                          result.prediction === 1 ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'
                        )}>
                          {result.prediction === 1 ? 'Malicious' : 'Benign'}
                        </span>
                      </td>
                      <td className="py-2 text-dark-300 font-mono">
                        {result.probability.toFixed(4)}
                        <div className="w-32 h-1.5 bg-dark-700 rounded-full mt-1 overflow-hidden">
                          <div
                            className={clsx('h-full rounded-full transition-all',
                              result.prediction === 1 ? 'bg-red-500' : 'bg-green-500'
                            )}
                            style={{ width: `${(result.probability * 100).toFixed(1)}%` }}
                          />
                        </div>
                      </td>
                      <td className="py-2 text-dark-300 font-mono">{result.confidence.toFixed(4)}</td>
                      <td className="py-2">
                        {result.true_label !== undefined ? (
                          <span className={clsx(
                            'px-2 py-0.5 text-xs rounded-full font-medium',
                            result.true_label === 1 ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'
                          )}>
                            {result.true_label === 1 ? 'Malicious' : 'Benign'}
                          </span>
                        ) : (
                          <span className="text-dark-500">N/A</span>
                        )}
                      </td>
                      <td className="py-2 text-dark-500 text-xs">
                        {result.timestamp ? format(new Date(result.timestamp), 'HH:mm:ss') : '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {totalPages > 1 && (
          <div className="p-4 border-t border-dark-700 flex items-center justify-center gap-2">
            <button
              onClick={() => setResultsPage(Math.max(1, resultsPage - 1))}
              disabled={resultsPage === 1}
              className="px-3 py-1.5 bg-dark-800 border border-dark-700 rounded-lg text-dark-300 hover:bg-dark-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="text-dark-400 px-4">
              Page {resultsPage} of {totalPages}
            </span>
            <button
              onClick={() => setResultsPage(Math.min(totalPages, resultsPage + 1))}
              disabled={resultsPage === totalPages}
              className="px-3 py-1.5 bg-dark-800 border border-dark-700 rounded-lg text-dark-300 hover:bg-dark-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export function Inference() {
  const { data: jobs, isLoading: jobsLoading, refetch: refetchJobs } = useInferenceJobs()

  const [isUploading, setIsUploading] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [jobName, setJobName] = useState('')

  const [selectedJob, setSelectedJob] = useState<InferenceJob | null>(null)
  const [jobResults, setJobResults] = useState<DetectionResult[]>([])
  const [isLoadingResults, setIsLoadingResults] = useState(false)
  const [showResultsModal, setShowResultsModal] = useState(false)
  const [resultsPage, setResultsPage] = useState(1)
  const [filterPrediction, setFilterPrediction] = useState<'all' | 'benign' | 'malicious'>('all')

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      if (!file.name.endsWith('.csv')) {
        alert('Please select a CSV file')
        return
      }
      setSelectedFile(file)
      if (!jobName) {
        setJobName(file.name.replace('.csv', ''))
      }
    }
  }

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedFile) {
      alert('Please select a CSV file')
      return
    }

    setIsUploading(true)
    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('name', jobName)

      await api.uploadCSVAndRunInference(formData)
      setSelectedFile(null)
      setJobName('')
      refetchJobs()
      alert('Inference completed successfully!')
    } catch (error: any) {
      console.error('Upload failed:', error)
      alert(`Upload failed: ${error.response?.data?.detail || error.message}`)
    } finally {
      setIsUploading(false)
    }
  }

  const handleViewResults = async (job: InferenceJob) => {
    setSelectedJob(job)
    setShowResultsModal(true)
    setResultsPage(1)
    setFilterPrediction('all')
    await loadResults(job.id)
  }

  const loadResults = async (jobId: string) => {
    setIsLoadingResults(true)
    try {
      const data = await api.getInferenceResults(jobId, { limit: 1000 })
      setJobResults(data.data)
    } catch (error) {
      console.error('Failed to load results:', error)
      setJobResults([])
    } finally {
      setIsLoadingResults(false)
    }
  }

  const filteredResults = jobResults.filter(r => {
    if (filterPrediction === 'all') return true
    if (filterPrediction === 'benign') return r.prediction === 0
    if (filterPrediction === 'malicious') return r.prediction === 1
    return true
  })

  const exportResults = () => {
    if (!selectedJob || jobResults.length === 0) return
    const headers = ['Record Index', 'Prediction', 'Probability', 'Confidence', 'True Label']
    const rows = jobResults.map(r => [
      r.record_index,
      r.prediction === 1 ? 'Malicious' : 'Benign',
      r.probability.toFixed(4),
      r.confidence.toFixed(4),
      r.true_label !== undefined ? (r.true_label === 1 ? 'Malicious' : 'Benign') : 'N/A'
    ])
    const csv = [headers, ...rows].map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `inference_results_${selectedJob.name}_${Date.now()}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-500/20 text-green-400'
      case 'running': return 'bg-blue-500/20 text-blue-400'
      case 'pending': return 'bg-yellow-500/20 text-yellow-400'
      case 'failed': return 'bg-red-500/20 text-red-400'
      default: return 'bg-gray-500/20 text-gray-400'
    }
  }

  return (
    <div className="space-y-6 animate-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-50">Inference Jobs</h1>
          <p className="text-dark-400 mt-1">Run model inference on datasets</p>
        </div>
        <button
          onClick={() => { setSelectedFile(null); setJobName(''); }}
          className="px-4 py-2 bg-primary-500 text-dark-950 rounded-lg font-medium hover:bg-primary-400 flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          New Inference Job
        </button>
      </div>

      <div className="p-6 rounded-xl border border-dark-700 bg-dark-900/50">
        <h3 className="text-lg font-semibold text-dark-50 mb-4 flex items-center gap-2">
          <Upload className="w-5 h-5 text-primary-400" />
          Run Inference on CSV Upload
        </h3>

        <form onSubmit={handleUpload} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-dark-300 mb-2">CSV File</label>
              <div className="relative">
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  disabled={isUploading}
                />
                <div className={clsx(
                  'border-2 border-dashed rounded-lg p-6 text-center transition-colors',
                  selectedFile ? 'border-green-500 bg-green-500/10' : 'border-dark-700 hover:border-primary-500'
                )}>
                  {selectedFile ? (
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <FileText className="w-8 h-8 text-green-400" />
                        <div>
                          <p className="font-medium text-dark-50">{selectedFile.name}</p>
                          <p className="text-sm text-dark-400">{(selectedFile.size / 1024).toFixed(1)} KB</p>
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => setSelectedFile(null)}
                        className="text-red-400 hover:text-red-300"
                      >
                        <X className="w-5 h-5" />
                      </button>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center gap-2">
                      <Upload className="w-10 h-10 text-dark-400" />
                      <p className="text-dark-300">Drag & drop or click to select CSV file</p>
                      <p className="text-sm text-dark-500">Must contain CIC-IDS2017 features (70 columns)</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-dark-300 mb-2">Job Name</label>
              <input
                type="text"
                value={jobName}
                onChange={(e) => setJobName(e.target.value)}
                placeholder="My inference job"
                disabled={isUploading}
                className="w-full px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 placeholder-dark-400 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={isUploading || !selectedFile}
            className={clsx(
              'w-full py-3 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors',
              isUploading || !selectedFile
                ? 'bg-dark-700 text-dark-400 cursor-not-allowed'
                : 'bg-primary-500 text-dark-950 hover:bg-primary-400'
            )}
          >
            {isUploading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Running Inference...
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Run Inference
              </>
            )}
          </button>
        </form>
      </div>

      <div className="p-6 rounded-xl border border-dark-700 bg-dark-900/50">
        <div className="flex gap-4 mb-6">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-400" />
            <input
              type="text"
              placeholder="Search inference jobs..."
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
                <th className="pb-3 font-medium text-dark-300">Model</th>
                <th className="pb-3 font-medium text-dark-300">Status</th>
                <th className="pb-3 font-medium text-dark-300">Records</th>
                <th className="pb-3 font-medium text-dark-300">Benign</th>
                <th className="pb-3 font-medium text-dark-300">Malicious</th>
                <th className="pb-3 font-medium text-dark-300">Created</th>
                <th className="pb-3 font-medium text-dark-300">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-800">
              {jobsLoading ? (
                <tr>
                  <td className="py-4 text-dark-400 text-center" colSpan={8}>
                    <div className="flex items-center justify-center gap-2">
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Loading...
                    </div>
                  </td>
                </tr>
              ) : jobs && jobs.length > 0 ? (
                jobs.map((job: InferenceJob) => (
                  <tr key={job.id} className="hover:bg-dark-800/50 transition-colors">
                    <td className="py-4 font-medium text-dark-50">{job.name}</td>
                    <td className="py-4 text-dark-400">{job.model_id.slice(0, 8)}...</td>
                    <td className="py-4">
                      <span className={clsx('px-2 py-1 text-xs rounded-full', getStatusColor(job.status))}>
                        {job.status}
                      </span>
                    </td>
                    <td className="py-4 text-dark-300">{job.total_records?.toLocaleString() ?? '-'}</td>
                    <td className="py-4 text-green-400">{job.benign_count?.toLocaleString() ?? '-'}</td>
                    <td className="py-4 text-red-400">{job.malicious_count?.toLocaleString() ?? '-'}</td>
                    <td className="py-4 text-dark-400">
                      {job.created_at ? format(new Date(job.created_at), 'MMM d, yyyy HH:mm') : '-'}
                    </td>
                    <td className="py-4">
                      <div className="flex items-center gap-2">
                        {job.status === 'completed' && (
                          <button
                            onClick={() => handleViewResults(job)}
                            className="px-3 py-1.5 text-sm bg-primary-500/10 text-primary-400 rounded-lg hover:bg-primary-500/20 flex items-center gap-1 transition-colors"
                          >
                            <FileText className="w-4 h-4" />
                            View Results
                          </button>
                        )}
                        {job.status === 'completed' && (
                          <button
                            onClick={() => {
                              setSelectedJob(job)
                              loadResults(job.id)
                              setShowResultsModal(true)
                              setResultsPage(1)
                              setFilterPrediction('all')
                            }}
                            className="px-3 py-1.5 text-sm bg-green-500/10 text-green-400 rounded-lg hover:bg-green-500/20 flex items-center gap-1 transition-colors"
                          >
                            <Download className="w-4 h-4" />
                            Export CSV
                          </button>
                        )}
                        <button
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
                  <td className="py-12 text-dark-400 text-center" colSpan={8}>
                    <div className="flex flex-col items-center gap-3">
                      <FileText className="w-12 h-12 text-dark-600" />
                      <p>No inference jobs yet</p>
                      <p className="text-sm">Upload a CSV file above to run inference</p>
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <ResultsModal
        isOpen={showResultsModal}
        onClose={() => setShowResultsModal(false)}
        job={selectedJob}
        results={jobResults}
        isLoading={isLoadingResults}
        filteredResults={filteredResults}
        resultsPage={resultsPage}
        setResultsPage={setResultsPage}
        filterPrediction={filterPrediction}
        setFilterPrediction={setFilterPrediction}
        exportResults={exportResults}
      />
    </div>
  )
}