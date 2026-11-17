'use client'

import { useState, useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import {
  Search, Filter, Upload, FileText, Download, Trash2,
  Loader2, X, ChevronLeft, ChevronRight, Database, CheckCircle,
  AlertCircle, AlertTriangle, Check,
  BarChart2, Zap, Shield, Eye, ArrowRight, ArrowLeft,
  RefreshCw, Hash, Table, Microscope
} from 'lucide-react'
import { clsx } from 'clsx'
import { useInferenceJobs } from '@/hooks/useDashboard'
import { api } from '@/services/api'
import { format } from 'date-fns'

// Types
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

interface DatasetAnalysis {
  dataset_id: string
  filename: string
  rows: number
  columns: number
  feature_names: string[]
  sample_rows: Record<string, any>[]
  column_stats: Record<string, Record<string, any>>
  label_distribution?: Record<string, number>
  has_label_column: boolean
}

interface DatasetPreview {
  dataset_id: string
  total_rows: number
  total_columns: number
  page: number
  page_size: number
  total_pages: number
  columns: string[]
  rows: Record<string, any>[]
}

interface ValidationCheck {
  name: string
  status: 'valid' | 'warning' | 'error'
  message: string
  details: Record<string, any>
}

interface ValidationReport {
  dataset_id: string
  is_valid: boolean
  status: 'valid' | 'warning' | 'error'
  checks: ValidationCheck[]
  feature_compatibility: Record<string, any>
  preprocessing_compatibility: boolean
}

interface CompatibleModel {
  id: string
  version: string
  name: string
  algorithm: string
  architecture: string
  metrics?: Record<string, any>
  threshold: number
  input_dim: number
}

// Sample datasets
const SAMPLE_DATASETS = [
  { id: 'batch_01', name: 'Sample Dataset', file: 'mock_inference_batch_01.csv', rows: 833 },
]

// Workflow steps (Model auto-selected)
const WORKFLOW_STEPS = [
  { id: 1, name: 'Upload', icon: Upload, description: 'Select or upload CSV dataset' },
  { id: 2, name: 'Analyze', icon: BarChart2, description: 'Preview dataset structure & statistics' },
  { id: 3, name: 'Validate', icon: Shield, description: 'Check compatibility with preprocessing pipeline' },
  { id: 4, name: 'Preprocess', icon: Hash, description: 'Apply feature scaling & transformation' },
  { id: 5, name: 'Inference', icon: Zap, description: 'Run model inference on dataset' },
  { id: 6, name: 'Results', icon: CheckCircle, description: 'View detection results & export' },
]

const getStatusColor = (status: string) => {
  switch (status) {
    case 'completed': return 'bg-green-500/20 text-green-400'
    case 'running': return 'bg-blue-500/20 text-blue-400'
    case 'pending': return 'bg-yellow-500/20 text-yellow-400'
    case 'failed': return 'bg-red-500/20 text-red-400'
    default: return 'bg-gray-500/20 text-gray-400'
  }
}

const getCheckStatusIcon = (status: string) => {
  switch (status) {
    case 'valid': return <Check className="w-5 h-5 text-green-400" />
    case 'warning': return <AlertTriangle className="w-5 h-5 text-yellow-400" />
    case 'error': return <AlertCircle className="w-5 h-5 text-red-400" />
    default: return <AlertCircle className="w-5 h-5 text-gray-400" />
  }
}

const getCheckStatusColor = (status: string) => {
  switch (status) {
    case 'valid': return 'border-green-500/50 bg-green-500/10 text-green-400'
    case 'warning': return 'border-yellow-500/50 bg-yellow-500/10 text-yellow-400'
    case 'error': return 'border-red-500/50 bg-red-500/10 text-red-400'
    default: return 'border-gray-500/50 bg-gray-500/10 text-gray-400'
  }
}

// Step 1: Upload Component
function UploadStep({
  onNext,
  onSampleSelect,
  selectedFile,
  setSelectedFile,
  jobName,
  setJobName,
  useSampleData,
  selectedSample,
  isUploading,
}: {
  onNext: () => void
  onSampleSelect: (sample: typeof SAMPLE_DATASETS[0]) => void
  selectedFile: File | null
  setSelectedFile: (file: File | null) => void
  jobName: string
  setJobName: (name: string) => void
  useSampleData: string | null
  selectedSample: typeof SAMPLE_DATASETS[0]
  isUploading: boolean
}) {
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

  return (
    <div className="space-y-6 animate-in">
      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
        <label className="block text-sm font-medium text-dark-300 mb-3 flex items-center gap-2">
          <Database className="w-4 h-4 text-primary-400" />
          Use Sample Dataset (Mock IDS Data)
        </label>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {SAMPLE_DATASETS.map((sample) => (
            <button
              key={sample.id}
              type="button"
              onClick={() => onSampleSelect(sample)}
              className={clsx(
                'p-3 rounded-lg border-2 text-left transition-all',
                useSampleData === sample.id
                  ? 'border-primary-500 bg-primary-500/10'
                  : 'border-dark-700 hover:border-primary-500/50'
              )}
            >
              <div className="flex items-center justify-between mb-1">
                <span className={clsx('font-medium text-sm', useSampleData === sample.id ? 'text-primary-400' : 'text-dark-50')}>
                  {sample.name}
                </span>
                {useSampleData === sample.id && <CheckCircle className="w-4 h-4 text-primary-400" />}
              </div>
              <p className="text-xs text-dark-400">{sample.rows.toLocaleString()} rows</p>
            </button>
          ))}
        </div>
        {useSampleData && (
          <p className="mt-2 text-sm text-green-400 flex items-center gap-1">
            <CheckCircle className="w-3 h-3" />
            Using {selectedSample.name} ({selectedSample.rows.toLocaleString()} rows)
          </p>
        )}
      </div>

      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
        <label className="block text-sm font-medium text-dark-300 mb-3 flex items-center gap-2">
          <FileText className="w-4 h-4 text-primary-400" />
          Or Upload Custom CSV
        </label>
        <div className="relative">
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            disabled={isUploading || !!useSampleData}
          />
          <div className={clsx(
            'border-2 border-dashed rounded-lg p-6 text-center transition-colors',
            selectedFile ? 'border-green-500 bg-green-500/10' :
            useSampleData ? 'border-dark-700 opacity-50' : 'border-dark-700 hover:border-primary-500'
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
                <p className="text-sm text-dark-500">Must contain IDS features (70 columns)</p>
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

      <button
        onClick={onNext}
        disabled={isUploading || (!selectedFile && !useSampleData)}
        className={clsx(
          'w-full py-3 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors',
          isUploading || (!selectedFile && !useSampleData)
            ? 'bg-dark-700 text-dark-400 cursor-not-allowed'
            : 'bg-primary-500 text-dark-950 hover:bg-primary-400'
        )}
      >
        {isUploading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Analyzing...
          </>
        ) : (
          <>
            <ArrowRight className="w-5 h-5" />
            Analyze Dataset
          </>
        )}
      </button>
    </div>
  )
}

// Step 2: Dataset Analysis Component
function AnalysisStep({
  analysis,
  isLoading,
  onNext,
  onBack,
}: {
  analysis: DatasetAnalysis | null
  isLoading: boolean
  onNext: () => void
  onBack: () => void
}) {
  const [previewPage, setPreviewPage] = useState(1)
  const [previewData, setPreviewData] = useState<DatasetPreview | null>(null)
  const [isPreviewLoading, setIsPreviewLoading] = useState(false)
  const [featurePage, setFeaturePage] = useState(1)
  const [featuresPerPage, setFeaturesPerPage] = useState(20)

  useEffect(() => {
    if (analysis) {
      fetchPreview(1)
    }
  }, [analysis])

  const fetchPreview = async (page: number) => {
    if (!analysis) return
    setIsPreviewLoading(true)
    try {
      const data = await api.previewDataset(analysis.dataset_id, { page, page_size: 50 })
      setPreviewData(data.data)
      setPreviewPage(page)
    } catch (error) {
      console.error('Failed to fetch preview:', error)
    } finally {
      setIsPreviewLoading(false)
    }
  }

  const totalFeaturePages = Math.ceil((analysis?.feature_names.length ?? 0) / featuresPerPage)
  const displayedFeatures = analysis
    ? analysis.feature_names.slice((featurePage - 1) * featuresPerPage, featurePage * featuresPerPage)
    : []

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-primary-400" />
      </div>
    )
  }

  if (!analysis) return null

  return (
    <div className="space-y-6 animate-in">
      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-dark-900/50 rounded-lg">
            <p className="text-3xl font-bold text-dark-50">{analysis.rows.toLocaleString()}</p>
            <p className="text-sm text-dark-400">Total Rows</p>
          </div>
          <div className="text-center p-4 bg-dark-900/50 rounded-lg">
            <p className="text-3xl font-bold text-dark-50">{analysis.columns}</p>
            <p className="text-sm text-dark-400">Features</p>
          </div>
          <div className="text-center p-4 bg-dark-900/50 rounded-lg">
            <p className="text-3xl font-bold text-dark-50">{analysis.has_label_column ? 'Yes' : 'No'}</p>
            <p className="text-sm text-dark-400">Label Column</p>
          </div>
          <div className="text-center p-4 bg-dark-900/50 rounded-lg">
            <p className="text-3xl font-bold text-dark-50">{analysis.filename}</p>
            <p className="text-sm text-dark-400">Filename</p>
          </div>
        </div>
      </div>

      {/* Column Statistics */}
      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-lg font-semibold text-dark-50 flex items-center gap-2">
            <Table className="w-5 h-5 text-primary-400" />
            Feature Statistics
          </h4>
          <div className="flex items-center gap-2">
            <select
              value={featuresPerPage}
              onChange={(e) => {
                setFeaturesPerPage(Number(e.target.value))
                setFeaturePage(1)
              }}
              className="px-2 py-1 bg-dark-900 border border-dark-700 rounded text-dark-50 text-sm"
            >
              <option value={20}>20 per page</option>
              <option value={50}>50 per page</option>
              <option value={100}>Show all ({analysis.feature_names.length})</option>
            </select>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-dark-900 border-b border-dark-700 z-10">
              <tr>
                <th className="pb-2 font-medium text-dark-300 text-left">Feature</th>
                <th className="pb-2 font-medium text-dark-300 text-left">Type</th>
                <th className="pb-2 font-medium text-dark-300 text-left">Min</th>
                <th className="pb-2 font-medium text-dark-300 text-left">Max</th>
                <th className="pb-2 font-medium text-dark-300 text-left">Mean</th>
                <th className="pb-2 font-medium text-dark-300 text-left">Std</th>
                <th className="pb-2 font-medium text-dark-300 text-left">Nulls</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-800">
              {displayedFeatures.map((feature) => {
                const stats = analysis.column_stats[feature]
                return (
                  <tr key={feature} className="hover:bg-dark-800/50">
                    <td className="py-2 text-dark-300 font-mono text-xs max-w-[200px] truncate" title={feature}>{feature}</td>
                    <td className="py-2 text-dark-400">{stats?.type ?? '-'}</td>
                    <td className="py-2 text-dark-400 font-mono">{stats?.min !== undefined ? stats.min.toFixed(2) : '-'}</td>
                    <td className="py-2 text-dark-400 font-mono">{stats?.max !== undefined ? stats.max.toFixed(2) : '-'}</td>
                    <td className="py-2 text-dark-400 font-mono">{stats?.mean !== undefined ? stats.mean.toFixed(2) : '-'}</td>
                    <td className="py-2 text-dark-400 font-mono">{stats?.std !== undefined ? stats.std.toFixed(2) : '-'}</td>
                    <td className="py-2 text-dark-400">{stats?.null_count ?? 0}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        {totalFeaturePages > 1 && (
          <div className="flex items-center justify-between mt-4 pt-4 border-t border-dark-700">
            <span className="text-sm text-dark-400">
              Page {featurePage} of {totalFeaturePages} · {analysis.feature_names.length} features total
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setFeaturePage(Math.max(1, featurePage - 1))}
                disabled={featurePage === 1}
                className="p-1.5 rounded hover:bg-dark-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                onClick={() => setFeaturePage(Math.min(totalFeaturePages, featurePage + 1))}
                disabled={featurePage === totalFeaturePages}
                className="p-1.5 rounded hover:bg-dark-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Data Preview */}
      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-lg font-semibold text-dark-50 flex items-center gap-2">
            <Eye className="w-5 h-5 text-primary-400" />
            Data Preview (First {previewData?.total_rows ?? analysis.rows} rows)
          </h4>
          <div className="flex items-center gap-2 text-sm text-dark-400">
            <span>Page {previewPage} of {previewData?.total_pages ?? 1}</span>
            <button
              onClick={() => fetchPreview(Math.max(1, previewPage - 1))}
              disabled={previewPage === 1 || isPreviewLoading}
              className="p-1.5 rounded hover:bg-dark-700 disabled:opacity-50"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => fetchPreview(Math.min(previewData?.total_pages ?? 1, previewPage + 1))}
              disabled={previewPage === (previewData?.total_pages ?? 1) || isPreviewLoading}
              className="p-1.5 rounded hover:bg-dark-700 disabled:opacity-50"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
        <div className="overflow-x-auto overflow-y-auto max-h-[400px]">
          <table className="w-full text-xs min-w-max">
            <thead className="sticky top-0 bg-dark-900 border-b border-dark-700 z-10">
              <tr>
                {previewData?.columns.map((col) => (
                  <th key={col} className="pb-2 font-medium text-dark-300 text-left px-2 min-w-[100px]">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-800">
              {isPreviewLoading ? (
                <tr>
                  <td colSpan={previewData?.columns.length ?? 1} className="py-8 text-center">
                    <Loader2 className="w-6 h-6 animate-spin text-primary-400 mx-auto" />
                  </td>
                </tr>
              ) : previewData?.rows.map((row, idx) => (
                <tr key={idx} className="hover:bg-dark-800/50">
                  {previewData?.columns.map((col) => (
                    <td key={col} className="py-2 text-dark-400 font-mono px-2">
                      {typeof row[col] === 'number' ? row[col].toFixed(2) : row[col] ?? '-'}
                    </td>
                  ))}
                </tr>
              ))}
              {(!previewData?.rows.length) && !isPreviewLoading && (
                <tr>
                  <td colSpan={previewData?.columns.length ?? 1} className="py-8 text-center text-dark-500">
                    No data to display
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="flex justify-between">
        <button onClick={onBack} className="px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg flex items-center gap-2 text-dark-300 hover:bg-dark-700">
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        <button onClick={onNext} className="px-4 py-2 bg-primary-500 text-dark-950 rounded-lg flex items-center gap-2 hover:bg-primary-400">
          Validate Dataset
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}

// Step 3: Validation Component
function ValidationStep({
  validation,
  isLoading,
  onNext,
  onBack,
}: {
  validation: ValidationReport | null
  isLoading: boolean
  onNext: () => void
  onBack: () => void
}) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-primary-400" />
      </div>
    )
  }

  if (!validation) return null

  const statusColors = {
    valid: 'bg-green-500/10 border-green-500/20',
    warning: 'bg-yellow-500/10 border-yellow-500/20',
    error: 'bg-red-500/10 border-red-500/20',
  }

  const statusIcons = {
    valid: <CheckCircle className="w-6 h-6 text-green-400" />,
    warning: <AlertTriangle className="w-6 h-6 text-yellow-400" />,
    error: <AlertCircle className="w-6 h-6 text-red-400" />,
  }

  const statusLabels = {
    valid: 'Valid - Ready for Inference',
    warning: 'Warning - Inference Possible with Caveats',
    error: 'Error - Cannot Run Inference',
  }

  return (
    <div className="space-y-6 animate-in">
      <div className={clsx('p-4 rounded-lg border', statusColors[validation.status])}>
        <div className="flex items-center gap-4">
          {statusIcons[validation.status]}
          <div>
            <h3 className="text-lg font-semibold text-dark-50">{statusLabels[validation.status]}</h3>
            <p className="text-sm text-dark-400">
              {validation.checks.filter(c => c.status === 'error').length} errors,
              {validation.checks.filter(c => c.status === 'warning').length} warnings,
              {validation.checks.filter(c => c.status === 'valid').length} passed
            </p>
          </div>
        </div>
      </div>

      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
        <h4 className="text-lg font-semibold text-dark-50 mb-4 flex items-center gap-2">
          <Microscope className="w-5 h-5 text-primary-400" />
          Validation Checks
        </h4>
        <div className="space-y-3">
          {validation.checks.map((check) => (
            <div key={check.name} className={clsx('p-4 rounded-lg border', getCheckStatusColor(check.status))}>
              <div className="flex items-start gap-3">
                <div className="flex-shrink-0 mt-0.5">
                  {getCheckStatusIcon(check.status)}
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <h5 className="font-medium text-dark-50">{check.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h5>
                    <span className={clsx('px-2 py-0.5 text-xs rounded-full font-medium',
                      check.status === 'valid' ? 'bg-green-500/20 text-green-400' :
                      check.status === 'warning' ? 'bg-yellow-500/20 text-yellow-400' :
                      'bg-red-500/20 text-red-400'
                    )}>
                      {check.status.toUpperCase()}
                    </span>
                  </div>
                  <p className="text-sm text-dark-300 mt-1">{check.message}</p>
                  {Object.keys(check.details).length > 0 && (
                    <details className="mt-2">
                      <summary className="text-xs text-dark-400 cursor-pointer">Show details</summary>
                      <pre className="mt-2 p-2 bg-dark-900 rounded text-xs text-dark-300 overflow-x-auto">
                        {JSON.stringify(check.details, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Feature Compatibility */}
      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
        <h4 className="text-lg font-semibold text-dark-50 mb-4 flex items-center gap-2">
          <Hash className="w-5 h-5 text-primary-400" />
          Feature Compatibility
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-dark-400 mb-1">Expected Features</p>
            <p className="text-2xl font-bold text-dark-50">{validation.feature_compatibility.expected_features}</p>
          </div>
          <div>
            <p className="text-sm text-dark-400 mb-1">Actual Features</p>
            <p className="text-2xl font-bold text-dark-50">{validation.feature_compatibility.actual_features}</p>
          </div>
        </div>
        {validation.feature_compatibility.missing_features.length > 0 && (
          <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <p className="text-sm text-red-400 font-medium mb-1">Missing Features:</p>
            <p className="text-xs text-dark-300">{validation.feature_compatibility.missing_features.join(', ')}</p>
          </div>
        )}
        {validation.feature_compatibility.extra_features.length > 0 && (
          <div className="mt-4 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
            <p className="text-sm text-yellow-400 font-medium mb-1">Extra Features (will be ignored):</p>
            <p className="text-xs text-dark-300">{validation.feature_compatibility.extra_features.join(', ')}</p>
          </div>
        )}
      </div>

      <div className="flex justify-between">
        <button onClick={onBack} className="px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg flex items-center gap-2 text-dark-300 hover:bg-dark-700">
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        <button
          onClick={onNext}
          disabled={!validation.is_valid}
          className={clsx(
            'px-4 py-2 rounded-lg flex items-center gap-2',
            validation.is_valid
              ? 'bg-primary-500 text-dark-950 hover:bg-primary-400'
              : 'bg-dark-700 text-dark-400 cursor-not-allowed'
          )}
        >
          Run Inference
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}

// Step 4: Preprocessing Component
function PreprocessingStep({
  analysis,
  onNext,
  onBack,
}: {
  analysis: DatasetAnalysis | null
  onNext: () => void
  onBack: () => void
}) {
  const [preprocessStatus, setPreprocessStatus] = useState<'idle' | 'running' | 'completed' | 'failed'>('idle')
  const [preprocessMessage, setPreprocessMessage] = useState('')
  const [featureStats, setFeatureStats] = useState<Record<string, any> | null>(null)

  useEffect(() => {
    if (preprocessStatus === 'idle') {
      runPreprocessing()
    }
  }, [preprocessStatus])

  const runPreprocessing = async () => {
    if (!analysis) return
    setPreprocessStatus('running')
    setPreprocessMessage('Loading preprocessing artifacts...')
    try {
      // Call backend to preprocess the dataset
      const data = await api.preprocessDataset(analysis.dataset_id)
      setFeatureStats(data.data)
      setPreprocessStatus('completed')
      setPreprocessMessage('Preprocessing completed successfully')
    } catch (error: any) {
      setPreprocessStatus('failed')
      setPreprocessMessage(error.response?.data?.detail || error.message || 'Preprocessing failed')
    }
  }

  if (preprocessStatus === 'running') {
    return (
      <div className="space-y-6 animate-in">
        <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-lg font-semibold text-dark-50 flex items-center gap-2">
              <Loader2 className="w-5 h-5 text-primary-400 animate-spin" />
              Running Preprocessing
            </h4>
            <span className="text-sm font-mono text-primary-400">Processing...</span>
          </div>
          <div className="w-full h-3 bg-dark-900 rounded-full overflow-hidden">
            <div className="h-full bg-primary-500 rounded-full animate-pulse" style={{ width: '100%' }} />
          </div>
          <p className="text-sm text-dark-400 mt-2">{preprocessMessage}</p>
        </div>
      </div>
    )
  }

  if (preprocessStatus === 'completed') {
    return (
      <div className="space-y-6 animate-in">
        <div className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
          <div className="flex items-center gap-4">
            <CheckCircle className="w-8 h-8 text-green-400" />
            <div>
              <h3 className="text-lg font-semibold text-dark-50">Preprocessing Completed</h3>
              <p className="text-sm text-dark-400">{preprocessMessage}</p>
            </div>
          </div>
        </div>

        {featureStats && (
          <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
            <h4 className="text-lg font-semibold text-dark-50 mb-4 flex items-center gap-2">
              <Hash className="w-5 h-5 text-primary-400" />
              Preprocessing Summary
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-3 bg-dark-900/50 rounded-lg">
                <p className="text-xs text-dark-400">Features Processed</p>
                <p className="font-mono text-dark-50">{featureStats.features_processed ?? analysis?.columns ?? '-'}</p>
              </div>
              <div className="p-3 bg-dark-900/50 rounded-lg">
                <p className="text-xs text-dark-400">Scaler Applied</p>
                <p className="font-mono text-dark-50">{featureStats.scaler_type ?? 'StandardScaler'}</p>
              </div>
              <div className="p-3 bg-dark-900/50 rounded-lg">
                <p className="text-xs text-dark-400">Constant Features Removed</p>
                <p className="font-mono text-dark-50">{featureStats.constant_features_removed ?? 0}</p>
              </div>
            </div>
          </div>
        )}

        <div className="flex justify-between">
          <button onClick={onBack} className="px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg flex items-center gap-2 text-dark-300 hover:bg-dark-700">
            <ArrowLeft className="w-4 h-4" />
            Back
          </button>
          <button onClick={onNext} className="px-4 py-2 bg-primary-500 text-dark-950 rounded-lg flex items-center gap-2 hover:bg-primary-400">
            Run Inference
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    )
  }

  if (preprocessStatus === 'failed') {
    return (
      <div className="space-y-6 animate-in">
        <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
          <div className="flex items-center gap-4">
            <AlertCircle className="w-8 h-8 text-red-400" />
            <div>
              <h3 className="text-lg font-semibold text-dark-50">Preprocessing Failed</h3>
              <p className="text-sm text-dark-400">{preprocessMessage}</p>
            </div>
          </div>
        </div>
        <div className="flex justify-center">
          <button onClick={() => { setPreprocessStatus('idle'); runPreprocessing() }} className="px-4 py-2 bg-primary-500 text-dark-950 rounded-lg flex items-center gap-2 hover:bg-primary-400">
            <RefreshCw className="w-4 h-4" />
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-in">
      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg text-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary-400 mx-auto mb-4" />
        <p className="text-dark-400">Initializing preprocessing...</p>
      </div>
    </div>
  )
}


// Step 5: Inference Progress Component
function InferenceProgressStep({
  jobId,
  onComplete,
  onBack,
}: {
  jobId: string | null
  onComplete: () => void
  onBack: () => void
}) {
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState<'running' | 'completed' | 'failed'>('running')
  const [message, setMessage] = useState('Initializing...')
  const [results, setResults] = useState<{ benign: number; malicious: number; total: number } | null>(null)

  useEffect(() => {
    if (!jobId) return

    // Polling for results
    const pollForResults = async () => {
      if (status !== 'running') return
      try {
        const job = await api.getInferenceJob(jobId)
        if (job.data.status === 'completed') {
          setStatus('completed')
          setProgress(100)
          setMessage('Inference completed!')
          const benign = job.data.benign_count ?? 0
          const malicious = job.data.malicious_count ?? 0
          setResults({ benign, malicious, total: benign + malicious })
        } else if (job.data.status === 'failed') {
          setStatus('failed')
          setMessage(job.data.error_message || 'Inference failed')
        }
      } catch (e) {
        console.error('Polling failed:', e)
      }
    }

    const pollInterval = setInterval(pollForResults, 2000)

    return () => {
      clearInterval(pollInterval)
    }
  }, [jobId, status])

  if (status === 'completed') {
    return (
      <div className="space-y-6 animate-in">
        <div className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
          <div className="flex items-center gap-4">
            <CheckCircle className="w-8 h-8 text-green-400" />
            <div>
              <h3 className="text-lg font-semibold text-dark-50">Inference Completed Successfully</h3>
              <p className="text-sm text-dark-400">
                Processed {results?.total.toLocaleString()} records •
                {results?.benign.toLocaleString()} benign •
                {results?.malicious.toLocaleString()} malicious
              </p>
            </div>
          </div>
        </div>

        <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-4 bg-dark-900/50 rounded-lg">
            <p className="text-3xl font-bold text-dark-50">{results?.total.toLocaleString()}</p>
            <p className="text-sm text-dark-400">Total Records</p>
          </div>
          <div className="text-center p-4 bg-dark-900/50 rounded-lg">
            <p className="text-3xl font-bold text-green-400">{results?.benign.toLocaleString()}</p>
            <p className="text-sm text-dark-400">Benign</p>
          </div>
          <div className="text-center p-4 bg-dark-900/50 rounded-lg">
            <p className="text-3xl font-bold text-red-400">{results?.malicious.toLocaleString()}</p>
            <p className="text-sm text-dark-400">Malicious</p>
          </div>
        </div>

        <div className="flex justify-between">
          <button onClick={onBack} className="px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg flex items-center gap-2 text-dark-300 hover:bg-dark-700">
            <ArrowLeft className="w-4 h-4" />
            Back
          </button>
          <button onClick={onComplete} className="px-4 py-2 bg-primary-500 text-dark-950 rounded-lg flex items-center gap-2 hover:bg-primary-400">
            View Results
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    )
  }

  if (status === 'failed') {
    return (
      <div className="space-y-6 animate-in">
        <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
          <div className="flex items-center gap-4">
            <AlertCircle className="w-8 h-8 text-red-400" />
            <div>
              <h3 className="text-lg font-semibold text-dark-50">Inference Failed</h3>
              <p className="text-sm text-dark-400">{message}</p>
            </div>
          </div>
        </div>
        <div className="flex justify-center">
          <button onClick={onBack} className="px-4 py-2 bg-primary-500 text-dark-950 rounded-lg flex items-center gap-2 hover:bg-primary-400">
            <RefreshCw className="w-4 h-4" />
            Try Again
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-in">
      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-lg font-semibold text-dark-50 flex items-center gap-2">
            <Zap className="w-5 h-5 text-primary-400" />
            Running Inference
          </h4>
          <span className="text-sm font-mono text-primary-400">{progress}%</span>
        </div>
        <div className="w-full h-3 bg-dark-900 rounded-full overflow-hidden">
          <div
            className="h-full bg-primary-500 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className="text-sm text-dark-400 mt-2">{message}</p>
      </div>

      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
        <h4 className="text-lg font-semibold text-dark-50 mb-4 flex items-center gap-2">
          <Loader2 className="w-5 h-5 text-primary-400 animate-spin" />
          Inference Details
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
          <div className="p-3 bg-dark-900/50 rounded-lg">
            <p className="text-xs text-dark-400">Job ID</p>
            <p className="font-mono text-xs text-dark-300">{jobId?.slice(0, 8)}...</p>
          </div>
          <div className="p-3 bg-dark-900/50 rounded-lg">
            <p className="text-xs text-dark-400">Status</p>
            <p className="font-mono text-xs text-blue-400">Running</p>
          </div>
          <div className="p-3 bg-dark-900/50 rounded-lg">
            <p className="text-xs text-dark-400">Progress</p>
            <p className="font-mono text-xs text-primary-400">{progress}%</p>
          </div>
        </div>
      </div>
    </div>
  )
}

// Step 6: Results Component
function ResultsStep({
  results,
  isLoading,
  onExport,
  onViewDetections,
}: {
  results: DetectionResult[]
  isLoading: boolean
  onExport: () => void
  onViewDetections: () => void
}) {
  const [filterPrediction, setFilterPrediction] = useState<'all' | 'benign' | 'malicious'>('all')
  const [resultsPage, setResultsPage] = useState(1)

  const filteredResults = results.filter(r => {
    if (filterPrediction === 'all') return true
    if (filterPrediction === 'benign') return r.prediction === 0
    if (filterPrediction === 'malicious') return r.prediction === 1
    return true
  })

  const totalPages = Math.ceil(filteredResults.length / 20)
  const paginatedResults = filteredResults.slice(
    (resultsPage - 1) * 20,
    resultsPage * 20
  )

  const benignCount = results.filter(r => r.prediction === 0).length
  const maliciousCount = results.filter(r => r.prediction === 1).length

  return (
    <div className="space-y-6 animate-in">
      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="text-center">
          <p className="text-3xl font-bold text-dark-50">{results.length.toLocaleString()}</p>
          <p className="text-xs text-dark-400">Total Records</p>
        </div>
        <div className="text-center">
          <p className="text-3xl font-bold text-green-400">{benignCount.toLocaleString()}</p>
          <p className="text-xs text-dark-400">Benign</p>
        </div>
        <div className="text-center">
          <p className="text-3xl font-bold text-red-400">{maliciousCount.toLocaleString()}</p>
          <p className="text-xs text-dark-400">Malicious</p>
        </div>
        <div className="text-center">
          <p className="text-3xl font-bold text-primary-400">
            {(maliciousCount / Math.max(results.length, 1) * 100).toFixed(1)}%
          </p>
          <p className="text-xs text-dark-400">Malicious Rate</p>
        </div>
      </div>

      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg flex flex-wrap items-center gap-4">
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
          <button onClick={onExport} className="px-4 py-2 bg-green-500/10 text-green-400 rounded-lg hover:bg-green-500/20 flex items-center gap-2 transition-colors text-sm">
            <Download className="w-4 h-4" />
            Export CSV
          </button>
          <button onClick={onViewDetections} className="px-4 py-2 bg-primary-500/10 text-primary-400 rounded-lg hover:bg-primary-500/20 flex items-center gap-2 transition-colors text-sm">
            <BarChart2 className="w-4 h-4" />
            View in Detections
          </button>
        </div>
      </div>

      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
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

        {totalPages > 1 && (
          <div className="mt-4 flex items-center justify-center gap-2">
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

// Main Inference Component
export function Inference() {
  const queryClient = useQueryClient()
  const { data: jobs, isLoading: jobsLoading } = useInferenceJobs()

  // Workflow state
  const [currentStep, setCurrentStep] = useState(1)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [jobName, setJobName] = useState('')
  const [useSampleData, setUseSampleData] = useState<string | null>(null)
  const [selectedSample, setSelectedSample] = useState<typeof SAMPLE_DATASETS[0]>(SAMPLE_DATASETS[0])
  const [analysis, setAnalysis] = useState<DatasetAnalysis | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [validation, setValidation] = useState<ValidationReport | null>(null)
  const [isValidating, setIsValidating] = useState(false)
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null)

  // Delete inference job
  const handleDeleteJob = async (jobId: string) => {
    if (!confirm('Delete this inference job and all its results?')) return
    try {
      await api.deleteInferenceJob(jobId)
      queryClient.invalidateQueries({ queryKey: ['inferenceJobs'] })
    } catch (error: any) {
      alert(`Failed to delete: ${error.response?.data?.detail || error.message}`)
    }
  }
  const [inferenceJobId, setInferenceJobId] = useState<string | null>(null)
  const [jobResults, setJobResults] = useState<DetectionResult[]>([])
  const [isLoadingResults, setIsLoadingResults] = useState(false)
  const [completedJob, setCompletedJob] = useState<InferenceJob | null>(null)

  const handleSampleSelect = (sample: typeof SAMPLE_DATASETS[0]) => {
    setSelectedSample(sample)
    setUseSampleData(sample.id)
    setSelectedFile(null)
    setJobName(sample.name)
  }

  // Step 1 -> 2: Analyze CSV
  const handleAnalyze = async () => {
    setIsAnalyzing(true)
    try {
      if (useSampleData) {
        const data = await api.analyzeSampleDataset(selectedSample.file)
        setAnalysis(data.data)
      } else if (selectedFile) {
        const formData = new FormData()
        formData.append('file', selectedFile)
        const data = await api.analyzeCSVDataset(formData)
        setAnalysis(data.data)
      }
      setCurrentStep(2)
    } catch (error: any) {
      console.error('Analysis failed:', error)
      alert(`Analysis failed: ${error.response?.data?.detail || error.message}`)
    } finally {
      setIsAnalyzing(false)
    }
  }

  // Step 2 -> 3: Validate Dataset
  const handleValidate = async () => {
    if (!analysis) return
    setIsValidating(true)
    try {
      const data = await api.validateDataset(analysis.dataset_id)
      setValidation(data.data)
      setCurrentStep(3)
    } catch (error: any) {
      console.error('Validation failed:', error)
      alert(`Validation failed: ${error.response?.data?.detail || error.message}`)
    } finally {
      setIsValidating(false)
    }
  }

  // Step 3 -> 4: Load Models & Auto-select Best
  const handleLoadModels = async () => {
    if (!analysis) return
        try {
      const data = await api.listCompatibleModels(analysis.dataset_id)
      const models = data.data
      
      // Auto-select best model (highest accuracy)
      const bestModel = models.reduce((best: CompatibleModel, current: CompatibleModel) => {
        const bestAcc = best.metrics?.accuracy ?? 0
        const currentAcc = current.metrics?.accuracy ?? 0
        return currentAcc > bestAcc ? current : best
      }, models[0])

      if (bestModel) {
        setSelectedModelId(bestModel.id)
        // Skip model selection step, go directly to inference
        handleRunInference()
      } else {
        // No models available, go back to validate step
        setCurrentStep(3)
      }
    } catch (error: any) {
      console.error('Failed to load models:', error)
      alert(`Failed to load models: ${error.response?.data?.detail || error.message}`)
    } finally {
          }
  }

  // Step 4 -> 5: Run Inference
  const handleRunInference = async () => {
    if (!analysis || !selectedModelId) return
        try {
      if (useSampleData) {
        const data = await api.uploadSampleAndRunInference({
          model_id: selectedModelId,
          name: jobName,
          sample_file: selectedSample.file,
          threshold: 0.5,
          batch_size: 256,
        })
        setInferenceJobId(data.data.id)
      } else if (selectedFile && analysis?.dataset_id) {
        // Use the existing analyzed/validated/preprocessed dataset
        const data = await api.runInferenceOnDataset({
          dataset_id: analysis.dataset_id,
          model_id: selectedModelId,
          name: jobName,
          threshold: 0.5,
          batch_size: 256,
        })
        setInferenceJobId(data.data.id)
      } else if (selectedFile) {
        // Fallback: upload file directly (shouldn't happen in normal flow)
        const formData = new FormData()
        formData.append('file', selectedFile)
        formData.append('name', jobName)
        formData.append('model_id', selectedModelId)
        const data = await api.uploadCSVAndRunInference(formData)
        setInferenceJobId(data.data.id)
      }
      setCurrentStep(5)
    } catch (error: any) {
      console.error('Inference failed:', error)
      alert(`Inference failed: ${error.response?.data?.detail || error.message}`)
          }
  }

  // Step 6 -> 7: Load Results
  const handleLoadResults = async () => {
    if (!inferenceJobId) return
    setIsLoadingResults(true)
    try {
      const [jobRes, resultsRes] = await Promise.all([
        api.getInferenceJob(inferenceJobId),
        api.getInferenceResults(inferenceJobId, { limit: 10000 }),
      ])
      setCompletedJob(jobRes.data)
      setJobResults(resultsRes.data)
      setCurrentStep(6)
    } catch (error: any) {
      console.error('Failed to load results:', error)
      alert(`Failed to load results: ${error.response?.data?.detail || error.message}`)
    } finally {
      setIsLoadingResults(false)
    }
  }

  const handleExport = () => {
    if (!completedJob || jobResults.length === 0) return
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
    a.download = `inference_results_${completedJob.name}_${Date.now()}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleViewDetections = () => {
    // Navigate to Detections page - this would typically use router
    window.location.href = '/detections'
  }

  const handleRestart = () => {
    setCurrentStep(1)
    setSelectedFile(null)
    setJobName('')
    setUseSampleData(null)
    setAnalysis(null)
    setValidation(null)
        setSelectedModelId(null)
    setInferenceJobId(null)
    setJobResults([])
    setCompletedJob(null)
  }

  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-50">Inference Workflow</h1>
          <p className="text-dark-400 mt-1">Run model inference on IDS datasets</p>
        </div>
        {currentStep > 1 && (
          <button
            onClick={handleRestart}
            className="px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg flex items-center gap-2 text-dark-300 hover:bg-dark-700"
          >
            <RefreshCw className="w-4 h-4" />
            New Job
          </button>
        )}
      </div>

      {/* Step Indicator */}
      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg overflow-x-auto">
        <div className="flex items-center gap-1 min-w-max">
          {WORKFLOW_STEPS.map((step, index) => {
            const isActive = currentStep === step.id
            const isCompleted = currentStep > step.id
            return (
              <div key={step.id} className="flex items-center">
                <div className={clsx(
                  'flex items-center gap-2 px-3 py-2 rounded-lg transition-all',
                  isActive ? 'bg-primary-500/10 border border-primary-500/30' :
                  isCompleted ? 'bg-green-500/10 border border-green-500/30' :
                  'bg-dark-900/50 border border-dark-700'
                )}>
                  {isCompleted ? (
                    <CheckCircle className={clsx('w-5 h-5', isCompleted ? 'text-green-400' : 'text-dark-400')} />
                  ) : (
                    <step.icon className={clsx('w-5 h-5', isActive ? 'text-primary-400' : 'text-dark-400')} />
                  )}
                  <span className={clsx('font-medium text-sm hidden sm:block', isActive ? 'text-primary-400' : isCompleted ? 'text-green-400' : 'text-dark-400')}>
                    {step.name}
                  </span>
                </div>
                {index < WORKFLOW_STEPS.length - 1 && (
                  <div className={clsx('w-8 h-0.5 mx-1', isCompleted ? 'bg-green-500' : 'bg-dark-700')} />
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Step Content */}
      <div className="p-6 bg-dark-900/50 border border-dark-700 rounded-xl min-h-[500px]">
        {currentStep === 1 && (
          <UploadStep
            onNext={handleAnalyze}
            onSampleSelect={handleSampleSelect}
            selectedFile={selectedFile}
            setSelectedFile={setSelectedFile}
            jobName={jobName}
            setJobName={setJobName}
            useSampleData={useSampleData}
            selectedSample={selectedSample}
            isUploading={isAnalyzing}
          />
        )}

        {currentStep === 2 && (
          <AnalysisStep
            analysis={analysis}
            isLoading={isAnalyzing}
            onNext={handleValidate}
            onBack={() => setCurrentStep(1)}
          />
        )}

        {currentStep === 3 && (
          <ValidationStep
            validation={validation}
            isLoading={isValidating}
            onNext={handleLoadModels}
            onBack={() => setCurrentStep(2)}
          />
        )}

        {currentStep === 4 && (
          <PreprocessingStep
            analysis={analysis}
            onNext={handleRunInference}
            onBack={() => setCurrentStep(3)}
          />
        )}

        {currentStep === 5 && (
          <InferenceProgressStep
            jobId={inferenceJobId}
            onComplete={handleLoadResults}
            onBack={() => setCurrentStep(4)}
          />
        )}

        {currentStep === 6 && (
          <ResultsStep
            results={jobResults}
            isLoading={isLoadingResults}
            onExport={handleExport}
            onViewDetections={handleViewDetections}
          />
        )}
      </div>

      {/* Recent Jobs Table */}
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
                  <td className="py-4 text-dark-400 text-center" colSpan={7}>
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
                            onClick={async () => {
                              setInferenceJobId(job.id)
                              await handleLoadResults()
                            }}
                            className="px-3 py-1.5 text-sm bg-primary-500/10 text-primary-400 rounded-lg hover:bg-primary-500/20 flex items-center gap-1 transition-colors"
                          >
                            <FileText className="w-4 h-4" />
                            View Results
                          </button>
                        )}
                        <button
                          onClick={() => handleDeleteJob(job.id)}
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
                  <td className="py-12 text-dark-400 text-center" colSpan={7}>
                    <div className="flex flex-col items-center gap-3">
                      <FileText className="w-12 h-12 text-dark-600" />
                      <p>No inference jobs yet</p>
                      <p className="text-sm">Complete the workflow above to run inference</p>
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}