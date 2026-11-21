'use client'

import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import {
  Search, Upload, FileText, Download, Trash2,
  Loader2, X, ChevronLeft, ChevronRight, Database, CheckCircle,
  AlertCircle, AlertTriangle, Check,
  BarChart2, Zap, Shield, Eye, ArrowRight, ArrowLeft,
  RefreshCw, Hash, Table, Sliders, Cpu, Activity, Info, ArrowUpDown
} from 'lucide-react'
import { clsx } from 'clsx'
import { useInferenceJobs } from '@/hooks/useDashboard'
import { api } from '@/services/api'
import { format } from 'date-fns'

// Types
interface TriggeredRule {
  rule_id: string
  name: string
  weight: number
  evidence: string
}

interface DetectionResult {
  id: string
  record_index: number
  prediction: number
  probability: number
  confidence: number
  true_label?: number | null
  features?: Record<string, any>
  timestamp: string
  predicted_category?: string
  supervised_probability?: number | null
  anomaly_score?: number | null
  rule_score?: number
  risk_score?: number
  severity?: string
  detection_mode?: string
  triggered_rules?: TriggeredRule[]
  explanation?: string
  top_anomaly_features?: string[]
  model_used?: string
  compatibility_score?: number
}

interface StageTelemetry {
  stage_id: string
  stage_name: string
  status: 'pending' | 'running' | 'completed' | 'warning' | 'failed'
  execution_time_ms: number
  input_count: number
  output_count: number
  warnings: string[]
  errors: string[]
  metadata: Record<string, any>
}

interface InferenceJob {
  id: string
  name: string
  model_id: string
  model_name?: string
  model_version?: string
  dataset_id: string
  organisation_id?: string
  status: string
  config?: Record<string, any>
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
  primary_label_column?: string | null
  file_size_bytes?: number
  memory_footprint_mb?: number
  detected_format?: string
  detected_encoding?: string
  detected_schema_family?: string
  schema_family_confidence?: number
  quality_score?: number
  quality_breakdown?: Record<string, number>
  valid_row_count?: number
  invalid_row_count?: number
  duplicate_rows?: number
  total_missing_cells?: number
  total_infinite_cells?: number
  column_types?: Record<string, string>
  type_counts?: Record<string, number>
  identifier_columns?: string[]
  constant_columns?: string[]
  network_related_columns?: string[]
  warnings?: string[]
}

interface DatasetPreview {
  dataset_id: string
  total_rows: number
  filtered_rows?: number
  total_columns: number
  page: number
  page_size: number
  total_pages: number
  columns: string[]
  column_types?: Record<string, string>
  missing_counts?: Record<string, number>
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
  status?: 'valid' | 'warning' | 'error'
  checks: ValidationCheck[]
  valid_rows?: number
  invalid_rows?: number
  duplicate_rows?: number
  missing_values_count?: number
  infinite_values_count?: number
  quality_score?: number
  detected_schema_family?: string
  selected_detection_mode?: string
  compatibility_score?: number
  feature_compatibility?: {
    expected_features: number
    actual_features: number
    mapped_features?: number
    exact_matches?: number
    semantic_matches?: number
    derived_features?: number
    approximate_matches?: number
    missing_features: string[]
    extra_features: string[]
    effective_coverage_ratio?: number
    core_coverage_ratio?: number
  }
  preprocessing_compatibility?: boolean
}

interface FeatureMappingEntry {
  canonical_id: string
  canonical_name: string
  cicids_name: string | null
  cicids_index: number | null
  group: string
  is_core: boolean
  source_column: string | null
  mapping_type: string
  confidence: number
  status: string
  unit_conversion?: string | null
  derivation_formula?: string | null
  explanation: string
}

interface FeatureMappingPayload {
  dataset_id: string
  uploaded_columns: string[]
  manual_overrides: Record<string, string>
  mapping_report: {
    schema_family: string
    total_uploaded_columns: number
    total_cicids_features: number
    mapped_cicids_count: number
    exact_match_count: number
    semantic_match_count: number
    derived_feature_count: number
    approximate_match_count: number
    ambiguous_match_count: number
    unavailable_feature_count: number
    incompatible_feature_count: number
    direct_coverage_ratio: number
    effective_coverage_ratio: number
    core_coverage_ratio: number
    average_mapping_confidence: number
    mappings: FeatureMappingEntry[]
    unmapped_uploaded_columns: string[]
    warnings: string[]
  }
  derivation_summary: {
    derived_features_count: number
    derived_features_list: string[]
    available_canonical_count: number
    missing_cicids_count: number
  }
  compatibility: {
    selected_mode: string
    selected_model_family: string
    selected_model_key: string
    selected_model_name: string
    is_supervised_compatible: boolean
    compatibility_score: number
    feature_coverage_score: number
    core_feature_coverage_score: number
    mapping_confidence_score: number
    data_quality_score: number
    distribution_compatibility_score: number
    available_modes: Array<{
      mode: string
      name: string
      compatible: boolean
      recommended: boolean
      description: string
    }>
    routing_reasons: string[]
    limitations: string[]
    scoring_weights: Record<string, number>
    missing_core_features: string[]
    missing_non_core_features: string[]
    derived_features_used: string[]
  }
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

// Sample datasets (Preserved mock datasets + Latest Kaggle datasets)
const SAMPLE_DATASETS = [
  { id: 'kaggle_intrusion_2025', name: 'Kaggle Cybersecurity Intrusion 2025', file: 'kaggle_cybersecurity_intrusion_2025.csv', rows: 9537, badge: 'Kaggle 2025' },
  { id: 'kaggle_unsw_nb15', name: 'Kaggle UNSW-NB15 Cloud IDS 2025', file: 'kaggle_unsw_nb15_cloud_ids_2025.csv', rows: 2000, badge: 'UNSW-NB15' },
  { id: 'kaggle_threat_intel', name: 'Kaggle Threat Intel Malicious IPs', file: 'kaggle_threat_intel_malicious_ips.csv', rows: 200, badge: 'Threat Intel' },
  { id: 'batch_01', name: 'CIC-IDS2017 Batch 01 (Unlabeled)', file: 'mock_inference_batch_01.csv', rows: 833, badge: 'CICIDS2017' },
  { id: 'batch_02', name: 'CIC-IDS2017 Batch 02 (Unlabeled)', file: 'mock_inference_batch_02.csv', rows: 833, badge: 'CICIDS2017' },
  { id: 'labelled_5000', name: 'CIC-IDS2017 Labeled (5,000 rows)', file: 'mock_cic_ids_5000_labelled.csv', rows: 5000, badge: 'Labeled' },
  { id: 'anon_100', name: 'Anonymized Feature Matrix (feature_1..70)', file: 'test_inference.csv', rows: 100, badge: 'Custom Schema' },
]

const WORKFLOW_STEPS = [
  { id: 1, name: 'Upload', icon: Upload, description: 'Upload CSV/TSV/JSON/Parquet or select dataset' },
  { id: 2, name: 'Overview & Preview', icon: BarChart2, description: 'Schema family, quality score & interactive preview' },
  { id: 3, name: 'Validate & Map', icon: Shield, description: 'Validation & semantic feature mapping table' },
  { id: 4, name: 'Model Routing', icon: Cpu, description: 'Adaptive mode selection & feature derivation' },
  { id: 5, name: '18-Stage Pipeline', icon: Zap, description: 'Live traceable stage-by-stage execution' },
  { id: 6, name: 'Detection Analytics', icon: CheckCircle, description: 'Risk, severity, explanations & investigation' },
]

const getStatusColor = (status: string) => {
  switch (status) {
    case 'completed': return 'bg-green-500/20 text-green-400'
    case 'warning': return 'bg-yellow-500/20 text-yellow-400'
    case 'running': return 'bg-blue-500/20 text-blue-400'
    case 'pending': return 'bg-dark-700 text-dark-400'
    case 'failed': return 'bg-red-500/20 text-red-400'
    default: return 'bg-gray-500/20 text-gray-400'
  }
}

const getSeverityBadge = (severity?: string) => {
  switch ((severity || 'informational').toLowerCase()) {
    case 'critical': return 'bg-red-500/25 text-red-300 border border-red-500/40'
    case 'high': return 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
    case 'medium': return 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
    case 'low': return 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
    default: return 'bg-dark-700 text-dark-300 border border-dark-600'
  }
}

const getMappingBadge = (mappingType: string) => {
  switch (mappingType) {
    case 'exact_match': return { label: 'Exact Match', cls: 'bg-green-500/20 text-green-400 border-green-500/30' }
    case 'strong_semantic_match': return { label: 'Semantic Match', cls: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' }
    case 'derived_feature': return { label: 'Derived Feature', cls: 'bg-blue-500/20 text-blue-400 border-blue-500/30' }
    case 'approximate_match': return { label: 'Approximate', cls: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30' }
    case 'ambiguous_match': return { label: 'Ambiguous', cls: 'bg-orange-500/20 text-orange-400 border-orange-500/30' }
    case 'incompatible_feature': return { label: 'Incompatible', cls: 'bg-red-500/20 text-red-400 border-red-500/30' }
    default: return { label: 'Unavailable', cls: 'bg-dark-700 text-dark-400 border-dark-600' }
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
    case 'valid': return 'border-green-500/40 bg-green-500/10'
    case 'warning': return 'border-yellow-500/40 bg-yellow-500/10'
    case 'error': return 'border-red-500/40 bg-red-500/10'
    default: return 'border-gray-500/40 bg-gray-500/10'
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
  isUploading,
}: {
  onNext: () => void
  onSampleSelect: (sample: typeof SAMPLE_DATASETS[0]) => void
  selectedFile: File | null
  setSelectedFile: (file: File | null) => void
  jobName: string
  setJobName: (name: string) => void
  useSampleData: string | null
  selectedSample?: typeof SAMPLE_DATASETS[0]
  isUploading: boolean
}) {
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      if (!jobName) {
        setJobName(file.name.replace(/\.[^/.]+$/, ''))
      }
    }
  }

  return (
    <div className="space-y-6 animate-in">
      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
        <label className="block text-sm font-medium text-dark-300 mb-3 flex items-center gap-2">
          <Database className="w-4 h-4 text-primary-400" />
          Select Built-in Benchmark or Synthetic Dataset
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
                <span className="text-xs px-2 py-0.5 rounded bg-dark-700 text-primary-400 font-mono">
                  {sample.badge}
                </span>
                {useSampleData === sample.id && <CheckCircle className="w-4 h-4 text-primary-400" />}
              </div>
              <p className={clsx('font-medium text-sm mt-1', useSampleData === sample.id ? 'text-primary-400' : 'text-dark-50')}>
                {sample.name}
              </p>
              <p className="text-xs text-dark-400 mt-1">{sample.rows.toLocaleString()} rows</p>
            </button>
          ))}
        </div>
      </div>

      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
        <label className="block text-sm font-medium text-dark-300 mb-3 flex items-center gap-2">
          <FileText className="w-4 h-4 text-primary-400" />
          Or Upload Any Network Traffic Dataset (CSV, TSV, JSON, JSONL, Parquet)
        </label>
        <div className="relative">
          <input
            type="file"
            accept=".csv,.tsv,.txt,.json,.jsonl,.ndjson,.parquet,.pq"
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
                <div className="flex items-center gap-3 text-left">
                  <FileText className="w-8 h-8 text-green-400" />
                  <div>
                    <p className="font-medium text-dark-50">{selectedFile.name}</p>
                    <p className="text-sm text-dark-400">{(selectedFile.size / 1024).toFixed(1)} KB</p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); setSelectedFile(null) }}
                  className="text-red-400 hover:text-red-300 z-10 relative"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2">
                <Upload className="w-10 h-10 text-dark-400" />
                <p className="text-dark-200 font-medium">Drag & drop or click to upload dataset</p>
                <p className="text-xs text-dark-400">
                  Supports CIC-IDS2017/2018/2019, UNSW-NB15, NSL-KDD, TON_IoT, Edge-IIoT, Zeek, NetFlow, and custom schemas
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-dark-300 mb-2">Analysis / Job Name</label>
        <input
          type="text"
          value={jobName}
          onChange={(e) => setJobName(e.target.value)}
          placeholder="Network traffic analysis run"
          disabled={isUploading}
          className="w-full px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-50 placeholder-dark-400 focus:border-primary-500 focus:outline-none"
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
            Ingesting & Profiling Dataset...
          </>
        ) : (
          <>
            <ArrowRight className="w-5 h-5" />
            Ingest & Profile Dataset
          </>
        )}
      </button>
    </div>
  )
}

// Step 2: Dataset Overview & Interactive Preview
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
  const [previewSearch, setPreviewSearch] = useState('')
  const [sortBy, setSortBy] = useState<string>('')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc')
  const [previewData, setPreviewData] = useState<DatasetPreview | null>(null)
  const [isPreviewLoading, setIsPreviewLoading] = useState(false)
  const [featureSearch, setFeatureSearch] = useState('')
  const [featurePage, setFeaturePage] = useState(1)
  const [featuresPerPage, setFeaturesPerPage] = useState(20)

  useEffect(() => {
    if (analysis) {
      fetchPreview(1, previewSearch, sortBy, sortOrder)
    }
  }, [analysis])

  const fetchPreview = async (page: number, search = previewSearch, sortCol = sortBy, order = sortOrder) => {
    if (!analysis) return
    setIsPreviewLoading(true)
    try {
      const data = await api.previewDataset(analysis.dataset_id, {
        page,
        page_size: 25,
        search: search || undefined,
        sort_by: sortCol || undefined,
        sort_order: order,
      })
      setPreviewData(data.data)
      setPreviewPage(page)
    } catch (error) {
      console.error('Failed to fetch preview:', error)
    } finally {
      setIsPreviewLoading(false)
    }
  }

  const handleSortColumn = (col: string) => {
    const nextOrder = sortBy === col && sortOrder === 'asc' ? 'desc' : 'asc'
    setSortBy(col)
    setSortOrder(nextOrder)
    fetchPreview(1, previewSearch, col, nextOrder)
  }

  if (isLoading || !analysis) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-primary-400" />
      </div>
    )
  }

  const filteredFeatures = analysis.feature_names.filter(f =>
    !featureSearch || f.toLowerCase().includes(featureSearch.toLowerCase())
  )
  const totalFeaturePages = Math.max(1, Math.ceil(filteredFeatures.length / featuresPerPage))
  const displayedFeatures = filteredFeatures.slice(
    (featurePage - 1) * featuresPerPage,
    featurePage * featuresPerPage
  )

  const fileSizeKB = analysis.file_size_bytes ? (analysis.file_size_bytes / 1024).toFixed(1) + ' KB' : '-'

  return (
    <div className="space-y-6 animate-in">
      {/* Dataset Overview Banner */}
      <div className="p-5 bg-dark-800/60 border border-dark-700 rounded-xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <span className="text-xs uppercase tracking-wider text-primary-400 font-semibold">
              Phase 3 — Universal Dataset Intelligence Profile
            </span>
            <h3 className="text-xl font-bold text-dark-50 mt-0.5">{analysis.filename}</h3>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="px-3 py-1 rounded-full text-xs font-semibold bg-primary-500/20 text-primary-400 border border-primary-500/30">
              Schema Family: {analysis.detected_schema_family || 'CUSTOM'}
            </span>
            <span className={clsx(
              'px-3 py-1 rounded-full text-xs font-semibold border',
              analysis.has_label_column
                ? 'bg-green-500/20 text-green-400 border-green-500/30'
                : 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
            )}>
              {analysis.has_label_column ? `Labeled (${analysis.primary_label_column || 'Label'})` : 'Unlabeled Dataset'}
            </span>
            <span className="px-3 py-1 rounded-full text-xs font-mono bg-dark-900 text-dark-300 border border-dark-700">
              Format: {(analysis.detected_format || 'csv').toUpperCase()} ({analysis.detected_encoding || 'utf-8'})
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <div className="p-3 bg-dark-900/60 rounded-lg border border-dark-700/60">
            <p className="text-xs text-dark-400">Records</p>
            <p className="text-2xl font-bold text-dark-50">{analysis.rows.toLocaleString()}</p>
            <p className="text-xs text-green-400 mt-0.5">{(analysis.valid_row_count ?? analysis.rows).toLocaleString()} clean rows</p>
          </div>
          <div className="p-3 bg-dark-900/60 rounded-lg border border-dark-700/60">
            <p className="text-xs text-dark-400">Columns</p>
            <p className="text-2xl font-bold text-dark-50">{analysis.columns}</p>
            <p className="text-xs text-dark-400 mt-0.5">{(analysis.network_related_columns?.length ?? 0)} network fields</p>
          </div>
          <div className="p-3 bg-dark-900/60 rounded-lg border border-dark-700/60">
            <p className="text-xs text-dark-400">File / Memory Size</p>
            <p className="text-xl font-bold text-dark-50">{fileSizeKB}</p>
            <p className="text-xs text-dark-400 mt-0.5">{analysis.memory_footprint_mb ?? 0} MB in RAM</p>
          </div>
          <div className="p-3 bg-dark-900/60 rounded-lg border border-dark-700/60">
            <p className="text-xs text-dark-400">Data Quality Score</p>
            <p className={clsx(
              'text-2xl font-bold',
              (analysis.quality_score ?? 90) >= 80 ? 'text-green-400' : (analysis.quality_score ?? 90) >= 60 ? 'text-yellow-400' : 'text-red-400'
            )}>
              {analysis.quality_score ?? 95}/100
            </p>
            <p className="text-xs text-dark-400 mt-0.5">Automated audit</p>
          </div>
          <div className="p-3 bg-dark-900/60 rounded-lg border border-dark-700/60">
            <p className="text-xs text-dark-400">Missing / Infinite</p>
            <p className="text-xl font-bold text-dark-50">
              {(analysis.total_missing_cells ?? 0).toLocaleString()} / {(analysis.total_infinite_cells ?? 0).toLocaleString()}
            </p>
            <p className="text-xs text-dark-400 mt-0.5">{(analysis.duplicate_rows ?? 0).toLocaleString()} duplicates</p>
          </div>
          <div className="p-3 bg-dark-900/60 rounded-lg border border-dark-700/60">
            <p className="text-xs text-dark-400">Column Types</p>
            <div className="text-xs text-dark-300 space-y-0.5 mt-1 font-mono">
              {Object.entries(analysis.type_counts || { numeric: analysis.columns }).map(([t, cnt]) => (
                <div key={t} className="flex justify-between">
                  <span className="capitalize">{t}:</span>
                  <span className="text-dark-50 font-semibold">{cnt}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Quality Breakdown Bars */}
        {analysis.quality_breakdown && (
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-2 border-t border-dark-700/60">
            {Object.entries(analysis.quality_breakdown).map(([metric, score]) => (
              <div key={metric} className="text-xs">
                <div className="flex justify-between text-dark-300 mb-1">
                  <span className="capitalize">{metric.replace(/_/g, ' ')}</span>
                  <span className="font-mono text-dark-50">{score}%</span>
                </div>
                <div className="w-full h-1.5 bg-dark-900 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary-500 rounded-full"
                    style={{ width: `${Math.min(100, Math.max(0, score))}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Interactive Data Preview */}
      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h4 className="text-lg font-semibold text-dark-50 flex items-center gap-2">
            <Eye className="w-5 h-5 text-primary-400" />
            Interactive Data Preview ({(previewData?.filtered_rows ?? analysis.rows).toLocaleString()} rows)
          </h4>
          <div className="flex flex-wrap items-center gap-2">
            <div className="relative">
              <Search className="w-4 h-4 text-dark-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={previewSearch}
                onChange={(e) => {
                  setPreviewSearch(e.target.value)
                  fetchPreview(1, e.target.value, sortBy, sortOrder)
                }}
                placeholder="Search rows..."
                className="pl-9 pr-3 py-1.5 bg-dark-900 border border-dark-700 rounded-lg text-sm text-dark-50 placeholder-dark-400 focus:border-primary-500 focus:outline-none"
              />
            </div>
            <div className="flex items-center gap-2 text-sm text-dark-400">
              <span>Page {previewPage} of {previewData?.total_pages ?? 1}</span>
              <button
                onClick={() => fetchPreview(Math.max(1, previewPage - 1))}
                disabled={previewPage === 1 || isPreviewLoading}
                className="p-1.5 rounded bg-dark-900 hover:bg-dark-700 disabled:opacity-50"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                onClick={() => fetchPreview(Math.min(previewData?.total_pages ?? 1, previewPage + 1))}
                disabled={previewPage === (previewData?.total_pages ?? 1) || isPreviewLoading}
                className="p-1.5 rounded bg-dark-900 hover:bg-dark-700 disabled:opacity-50"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        <div className="overflow-x-auto overflow-y-auto max-h-[360px] border border-dark-700 rounded-lg">
          <table className="w-full text-xs min-w-max">
            <thead className="sticky top-0 bg-dark-900 border-b border-dark-700 z-10">
              <tr>
                {previewData?.columns.map((col) => {
                  const colType = previewData.column_types?.[col] || analysis.column_types?.[col] || 'numeric'
                  const missCnt = previewData.missing_counts?.[col] ?? 0
                  return (
                    <th
                      key={col}
                      onClick={() => handleSortColumn(col)}
                      className="py-2.5 font-medium text-dark-200 text-left px-3 min-w-[130px] cursor-pointer hover:bg-dark-800 select-none"
                    >
                      <div className="flex items-center justify-between gap-1">
                        <span className="truncate max-w-[150px]" title={col}>{col}</span>
                        <ArrowUpDown className="w-3 h-3 text-dark-400 flex-shrink-0" />
                      </div>
                      <div className="flex items-center gap-1.5 mt-1">
                        <span className="text-[10px] px-1.5 py-0.2 rounded bg-dark-800 text-primary-400 font-mono">
                          {colType}
                        </span>
                        {missCnt > 0 && (
                          <span className="text-[10px] px-1.5 py-0.2 rounded bg-yellow-500/20 text-yellow-400 font-mono">
                            {missCnt} null
                          </span>
                        )}
                      </div>
                    </th>
                  )
                })}
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
                  {previewData.columns.map((col) => (
                    <td key={col} className="py-2 text-dark-300 font-mono px-3">
                      {typeof row[col] === 'number'
                        ? (Number.isInteger(row[col]) ? row[col] : row[col].toFixed(3))
                        : (row[col] !== null && row[col] !== undefined ? String(row[col]) : <span className="text-yellow-500/70">null</span>)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Column Statistics */}
      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h4 className="text-lg font-semibold text-dark-50 flex items-center gap-2">
            <Table className="w-5 h-5 text-primary-400" />
            Column & Distribution Statistics ({filteredFeatures.length} columns)
          </h4>
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={featureSearch}
              onChange={(e) => { setFeatureSearch(e.target.value); setFeaturePage(1) }}
              placeholder="Filter columns..."
              className="px-3 py-1 bg-dark-900 border border-dark-700 rounded text-sm text-dark-50 placeholder-dark-400"
            />
            <select
              value={featuresPerPage}
              onChange={(e) => { setFeaturesPerPage(Number(e.target.value)); setFeaturePage(1) }}
              className="px-2 py-1 bg-dark-900 border border-dark-700 rounded text-dark-50 text-sm"
            >
              <option value={20}>20 per page</option>
              <option value={50}>50 per page</option>
              <option value={100}>Show all</option>
            </select>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-dark-900 border-b border-dark-700">
              <tr>
                <th className="py-2 px-2 font-medium text-dark-300 text-left">Column</th>
                <th className="py-2 px-2 font-medium text-dark-300 text-left">Role / Type</th>
                <th className="py-2 px-2 font-medium text-dark-300 text-left">Min</th>
                <th className="py-2 px-2 font-medium text-dark-300 text-left">Median</th>
                <th className="py-2 px-2 font-medium text-dark-300 text-left">Max</th>
                <th className="py-2 px-2 font-medium text-dark-300 text-left">Mean ± Std</th>
                <th className="py-2 px-2 font-medium text-dark-300 text-left">Outliers</th>
                <th className="py-2 px-2 font-medium text-dark-300 text-left">Null / Inf</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-800">
              {displayedFeatures.map((feature) => {
                const stats = analysis.column_stats[feature] || {}
                return (
                  <tr key={feature} className="hover:bg-dark-800/50">
                    <td className="py-2 px-2 text-dark-200 font-mono text-xs max-w-[220px] truncate" title={feature}>{feature}</td>
                    <td className="py-2 px-2 text-xs">
                      <span className="px-2 py-0.5 rounded bg-dark-900 text-primary-400 font-mono">
                        {stats.semantic_type || stats.type || '-'}
                      </span>
                    </td>
                    <td className="py-2 px-2 text-dark-400 font-mono text-xs">{stats.min !== undefined && stats.min !== null ? Number(stats.min).toFixed(2) : '-'}</td>
                    <td className="py-2 px-2 text-dark-300 font-mono text-xs">{stats.median !== undefined && stats.median !== null ? Number(stats.median).toFixed(2) : '-'}</td>
                    <td className="py-2 px-2 text-dark-400 font-mono text-xs">{stats.max !== undefined && stats.max !== null ? Number(stats.max).toFixed(2) : '-'}</td>
                    <td className="py-2 px-2 text-dark-400 font-mono text-xs">
                      {stats.mean !== undefined && stats.mean !== null ? `${Number(stats.mean).toFixed(2)} ± ${Number(stats.std || 0).toFixed(2)}` : '-'}
                    </td>
                    <td className="py-2 px-2 text-dark-400 font-mono text-xs">
                      {stats.outlier_count !== undefined ? `${stats.outlier_count} (${stats.outlier_percentage || 0}%)` : '-'}
                    </td>
                    <td className="py-2 px-2 text-dark-400 font-mono text-xs">
                      {stats.null_count ?? 0} / {stats.inf_count ?? 0}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        {totalFeaturePages > 1 && (
          <div className="flex items-center justify-between mt-3 pt-3 border-t border-dark-700 text-sm text-dark-400">
            <span>Page {featurePage} of {totalFeaturePages}</span>
            <div className="flex gap-2">
              <button
                onClick={() => setFeaturePage(Math.max(1, featurePage - 1))}
                disabled={featurePage === 1}
                className="p-1.5 rounded bg-dark-900 hover:bg-dark-700 disabled:opacity-50"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                onClick={() => setFeaturePage(Math.min(totalFeaturePages, featurePage + 1))}
                disabled={featurePage === totalFeaturePages}
                className="p-1.5 rounded bg-dark-900 hover:bg-dark-700 disabled:opacity-50"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="flex justify-between">
        <button onClick={onBack} className="px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg flex items-center gap-2 text-dark-300 hover:bg-dark-700">
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        <button onClick={onNext} className="px-5 py-2.5 bg-primary-500 text-dark-950 font-medium rounded-lg flex items-center gap-2 hover:bg-primary-400">
          Validate & Map Features
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}

// Step 3: Validation & Visual Semantic Feature Mapping Component
function ValidationStep({
  validation,
  mappingData,
  isLoading,
  onSaveOverrides,
  onNext,
  onBack,
}: {
  validation: ValidationReport | null
  mappingData: FeatureMappingPayload | null
  isLoading: boolean
  onSaveOverrides: (overrides: Record<string, string>) => Promise<void>
  onNext: () => void
  onBack: () => void
}) {
  const [mappingFilter, setMappingFilter] = useState<string>('all')
  const [mappingSearch, setMappingSearch] = useState('')
  const [localOverrides, setLocalOverrides] = useState<Record<string, string>>({})
  const [isSavingOverrides, setIsSavingOverrides] = useState(false)

  useEffect(() => {
    if (mappingData?.manual_overrides) {
      setLocalOverrides(mappingData.manual_overrides)
    }
  }, [mappingData])

  if (isLoading || !validation) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-primary-400" />
      </div>
    )
  }

  const status = validation.status || (validation.is_valid ? 'valid' : 'error')
  const mp = mappingData?.mapping_report
  const allMappings = mp?.mappings || []

  const filteredMappings = allMappings.filter((m) => {
    if (mappingFilter !== 'all' && m.mapping_type !== mappingFilter) return false
    if (mappingSearch) {
      const q = mappingSearch.toLowerCase()
      return (
        m.canonical_name.toLowerCase().includes(q) ||
        (m.cicids_name || '').toLowerCase().includes(q) ||
        (m.source_column || '').toLowerCase().includes(q) ||
        m.group.toLowerCase().includes(q)
      )
    }
    return true
  })

  const handleApplyOverrides = async () => {
    setIsSavingOverrides(true)
    try {
      await onSaveOverrides(localOverrides)
    } finally {
      setIsSavingOverrides(false)
    }
  }

  return (
    <div className="space-y-6 animate-in">
      {/* Validation Banner */}
      <div className={clsx(
        'p-4 rounded-xl border flex flex-wrap items-center justify-between gap-4',
        status === 'valid' ? 'bg-green-500/10 border-green-500/30' :
        status === 'warning' ? 'bg-yellow-500/10 border-yellow-500/30' :
        'bg-red-500/10 border-red-500/30'
      )}>
        <div className="flex items-center gap-3">
          {getCheckStatusIcon(status)}
          <div>
            <h3 className="text-lg font-semibold text-dark-50">
              {status === 'valid' ? 'Dataset Validated — Full Supervised Compatibility' :
               status === 'warning' ? `Dataset Validated — Adaptive Routing Enabled (${(validation.selected_detection_mode || 'hybrid').toUpperCase()} Mode)` :
               'Validation Error — Cannot Run Inference'}
            </h3>
            <p className="text-sm text-dark-300">
              Schema Family: <span className="font-mono text-primary-400">{validation.detected_schema_family || 'CUSTOM'}</span> •
              Compatibility Score: <span className="font-mono text-dark-50">{validation.compatibility_score ?? 100}%</span> •
              Quality Score: <span className="font-mono text-dark-50">{validation.quality_score ?? 95}/100</span>
            </p>
          </div>
        </div>
        <div className="grid grid-cols-3 sm:grid-cols-5 gap-3 text-center text-xs">
          <div className="px-3 py-1.5 bg-dark-900/70 rounded-lg">
            <p className="text-dark-400">Valid Rows</p>
            <p className="font-bold text-green-400 text-sm">{(validation.valid_rows ?? 0).toLocaleString()}</p>
          </div>
          <div className="px-3 py-1.5 bg-dark-900/70 rounded-lg">
            <p className="text-dark-400">Invalid Rows</p>
            <p className="font-bold text-yellow-400 text-sm">{(validation.invalid_rows ?? 0).toLocaleString()}</p>
          </div>
          <div className="px-3 py-1.5 bg-dark-900/70 rounded-lg">
            <p className="text-dark-400">Missing Cells</p>
            <p className="font-bold text-dark-50 text-sm">{(validation.missing_values_count ?? 0).toLocaleString()}</p>
          </div>
          <div className="px-3 py-1.5 bg-dark-900/70 rounded-lg">
            <p className="text-dark-400">Duplicates</p>
            <p className="font-bold text-dark-50 text-sm">{(validation.duplicate_rows ?? 0).toLocaleString()}</p>
          </div>
          <div className="px-3 py-1.5 bg-dark-900/70 rounded-lg">
            <p className="text-dark-400">Infinite Cells</p>
            <p className="font-bold text-dark-50 text-sm">{(validation.infinite_values_count ?? 0).toLocaleString()}</p>
          </div>
        </div>
      </div>

      {/* Validation Checks Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {validation.checks.map((check) => (
          <div key={check.name} className={clsx('p-3.5 rounded-lg border', getCheckStatusColor(check.status))}>
            <div className="flex items-start gap-2.5">
              <div className="mt-0.5">{getCheckStatusIcon(check.status)}</div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <h5 className="font-medium text-sm text-dark-50 capitalize">
                    {check.name.replace(/_/g, ' ')}
                  </h5>
                  <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-dark-900/70 text-dark-200">
                    {check.status}
                  </span>
                </div>
                <p className="text-xs text-dark-300 mt-1">{check.message}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Phase 4: Visual Semantic Feature Mapping Table */}
      {mp && (
        <div className="p-5 bg-dark-800/50 border border-dark-700 rounded-xl space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h4 className="text-lg font-semibold text-dark-50 flex items-center gap-2">
                <Sliders className="w-5 h-5 text-primary-400" />
                Semantic Feature Compatibility & Derivation Mapping ({mp.mapped_cicids_count}/70 CIC-IDS2017 Features Covered)
              </h4>
              <p className="text-xs text-dark-400 mt-0.5">
                Effective Coverage: {(mp.effective_coverage_ratio * 100).toFixed(1)}% •
                Core Flow Coverage: {(mp.core_coverage_ratio * 100).toFixed(1)}% •
                Mean Confidence: {(mp.average_mapping_confidence * 100).toFixed(1)}%
              </p>
            </div>
            {Object.keys(localOverrides).length > 0 && (
              <button
                onClick={handleApplyOverrides}
                disabled={isSavingOverrides}
                className="px-3 py-1.5 bg-primary-500 text-dark-950 text-xs font-semibold rounded-lg hover:bg-primary-400 flex items-center gap-1.5"
              >
                {isSavingOverrides ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />}
                Apply Manual Mapping Overrides ({Object.keys(localOverrides).length})
              </button>
            )}
          </div>

          {/* Filter Pills */}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap gap-1.5 text-xs">
              {[
                { key: 'all', label: `All (${allMappings.length})` },
                { key: 'exact_match', label: `Exact (${mp.exact_match_count})` },
                { key: 'strong_semantic_match', label: `Semantic (${mp.semantic_match_count})` },
                { key: 'derived_feature', label: `Derived (${mp.derived_feature_count})` },
                { key: 'approximate_match', label: `Approximate (${mp.approximate_match_count})` },
                { key: 'unavailable_feature', label: `Unavailable (${mp.unavailable_feature_count})` },
              ].map((tab) => (
                <button
                  key={tab.key}
                  onClick={() => setMappingFilter(tab.key)}
                  className={clsx(
                    'px-2.5 py-1 rounded-lg font-medium transition-colors',
                    mappingFilter === tab.key
                      ? 'bg-primary-500 text-dark-950'
                      : 'bg-dark-900 text-dark-300 hover:bg-dark-700'
                  )}
                >
                  {tab.label}
                </button>
              ))}
            </div>
            <input
              type="text"
              value={mappingSearch}
              onChange={(e) => setMappingSearch(e.target.value)}
              placeholder="Search feature mappings..."
              className="px-3 py-1 bg-dark-900 border border-dark-700 rounded-lg text-xs text-dark-50 placeholder-dark-400"
            />
          </div>

          <div className="overflow-x-auto max-h-[420px] border border-dark-700 rounded-lg">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-dark-900 border-b border-dark-700 z-10">
                <tr>
                  <th className="py-2.5 px-3 text-left font-medium text-dark-300">Uploaded / Source Feature</th>
                  <th className="py-2.5 px-3 text-left font-medium text-dark-300">Canonical Feature (Model Target)</th>
                  <th className="py-2.5 px-3 text-left font-medium text-dark-300">Group</th>
                  <th className="py-2.5 px-3 text-left font-medium text-dark-300">Mapping Type</th>
                  <th className="py-2.5 px-3 text-left font-medium text-dark-300">Confidence</th>
                  <th className="py-2.5 px-3 text-left font-medium text-dark-300">Explanation & Provenance</th>
                  <th className="py-2.5 px-3 text-left font-medium text-dark-300">Manual Override</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-800">
                {filteredMappings.map((m) => {
                  const badge = getMappingBadge(m.mapping_type)
                  return (
                    <tr key={m.canonical_id} className="hover:bg-dark-800/40">
                      <td className="py-2 px-3 font-mono text-dark-200">
                        {m.source_column ? (
                          <span className="text-primary-300">{m.source_column}</span>
                        ) : (
                          <span className="text-dark-500 italic">— unmapped —</span>
                        )}
                      </td>
                      <td className="py-2 px-3">
                        <div className="font-medium text-dark-50 flex items-center gap-1.5">
                          <span>{m.cicids_name || m.canonical_name}</span>
                          {m.is_core && (
                            <span className="text-[10px] px-1.5 py-0.2 rounded bg-primary-500/20 text-primary-400 font-mono">
                              CORE
                            </span>
                          )}
                        </div>
                        <div className="font-mono text-[10px] text-dark-400">{m.canonical_id}</div>
                      </td>
                      <td className="py-2 px-3 text-dark-400 capitalize">{m.group.replace(/_/g, ' ')}</td>
                      <td className="py-2 px-3">
                        <span className={clsx('px-2 py-0.5 rounded-full text-[11px] border font-medium', badge.cls)}>
                          {badge.label}
                        </span>
                      </td>
                      <td className="py-2 px-3 font-mono text-dark-200">
                        {(m.confidence * 100).toFixed(0)}%
                      </td>
                      <td className="py-2 px-3 text-dark-300 max-w-[300px]" title={m.explanation}>
                        <p className="truncate">{m.explanation}</p>
                      </td>
                      <td className="py-2 px-3">
                        <select
                          value={localOverrides[m.canonical_id] || ''}
                          onChange={(e) => {
                            const val = e.target.value
                            setLocalOverrides((prev) => {
                              const next = { ...prev }
                              if (val) next[m.canonical_id] = val
                              else delete next[m.canonical_id]
                              return next
                            })
                          }}
                          className="px-2 py-1 bg-dark-900 border border-dark-700 rounded text-dark-200 text-xs max-w-[150px]"
                        >
                          <option value="">Auto ({m.status})</option>
                          {(mappingData?.uploaded_columns || []).map((uCol) => (
                            <option key={uCol} value={uCol}>{uCol}</option>
                          ))}
                        </select>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="flex justify-between">
        <button onClick={onBack} className="px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg flex items-center gap-2 text-dark-300 hover:bg-dark-700">
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        <button
          onClick={onNext}
          disabled={!validation.is_valid}
          className={clsx(
            'px-5 py-2.5 rounded-lg font-medium flex items-center gap-2',
            validation.is_valid
              ? 'bg-primary-500 text-dark-950 hover:bg-primary-400'
              : 'bg-dark-700 text-dark-400 cursor-not-allowed'
          )}
        >
          Configure Model Routing & Preprocessing
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}

// Step 4: Model Compatibility, Adaptive Routing & Preprocessing Component
function PreprocessingStep({
  analysis,
  mappingData,
  compatibleModels,
  selectedModelId,
  setSelectedModelId,
  selectedMode,
  setSelectedMode,
  onNext,
  onBack,
}: {
  analysis: DatasetAnalysis | null
  mappingData: FeatureMappingPayload | null
  compatibleModels: CompatibleModel[]
  selectedModelId: string | null
  setSelectedModelId: (id: string) => void
  selectedMode: string
  setSelectedMode: (mode: string) => void
  onNext: () => void
  onBack: () => void
}) {
  const [preprocessStatus, setPreprocessStatus] = useState<'idle' | 'running' | 'completed' | 'failed'>('idle')
  const [preprocessMessage, setPreprocessMessage] = useState('')
  const [featureStats, setFeatureStats] = useState<Record<string, any> | null>(null)

  useEffect(() => {
    if (preprocessStatus === 'idle' && analysis) {
      runPreprocessing()
    }
  }, [preprocessStatus, analysis])

  const runPreprocessing = async () => {
    if (!analysis) return
    setPreprocessStatus('running')
    setPreprocessMessage('Executing safe feature derivations & adaptive scaling...')
    try {
      const data = await api.preprocessDataset(analysis.dataset_id)
      setFeatureStats(data.data)
      if (!selectedMode && data.data.selected_mode) {
        setSelectedMode(data.data.selected_mode)
      }
      setPreprocessStatus('completed')
      setPreprocessMessage('Feature derivation, routing & preprocessing verified')
    } catch (error: any) {
      setPreprocessStatus('failed')
      setPreprocessMessage(error.response?.data?.detail || error.message || 'Preprocessing failed')
    }
  }

  const compat = mappingData?.compatibility
  const activeModelObj = compatibleModels.find((m) => m.id === selectedModelId)
  const activeModeVal = selectedMode || compat?.selected_mode || 'supervised'
  const usesSupervisedCheckpoint = activeModeVal === 'supervised' || activeModeVal === 'hybrid'

  return (
    <div className="space-y-6 animate-in">
      {/* Trained Model Checkpoint Selector (Federated / DP-FedAvg / Centralized) */}
      {compatibleModels.length > 0 && (
        <div className="p-5 bg-dark-800/60 border border-dark-700 rounded-xl space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <span className="text-xs uppercase tracking-wider text-primary-400 font-semibold">
                Global & Federated Model Selection
              </span>
              <h3 className="text-lg font-bold text-dark-50 mt-0.5">
                Active Model Checkpoint: {activeModelObj ? activeModelObj.name : 'Federated v2 (Improved FedAvg, Round 20)'}
              </h3>
              <p className="text-xs text-dark-400 mt-0.5">
                {usesSupervisedCheckpoint
                  ? 'Select which Federated (FedAvg / DP-FedAvg) or Centralized model checkpoint to execute during Stage 11 inference.'
                  : 'Note: Current detection mode relies on Unsupervised Anomaly + Rule Engine; selected checkpoint is recorded for job lineage.'}
              </p>
            </div>
            {activeModelObj && (
              <div className="flex items-center gap-2 text-xs font-mono">
                <span className="px-2.5 py-1 rounded bg-primary-500/20 text-primary-400 border border-primary-500/30 uppercase">
                  {activeModelObj.algorithm}
                </span>
                <span className="px-2.5 py-1 rounded bg-dark-900 text-dark-200 border border-dark-700">
                  {activeModelObj.architecture}
                </span>
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {compatibleModels.map((m) => {
              const isSelected = m.id === selectedModelId
              const isFederated = m.algorithm.includes('fed')
              const isDP = m.algorithm.includes('dp')
              const acc = m.metrics?.accuracy !== undefined ? (Number(m.metrics.accuracy) * 100).toFixed(2) + '%' : 'N/A'
              const f1 = m.metrics?.f1 !== undefined ? (Number(m.metrics.f1) * 100).toFixed(2) + '%' : 'N/A'
              const recall = m.metrics?.recall !== undefined ? (Number(m.metrics.recall) * 100).toFixed(2) + '%' : 'N/A'
              return (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setSelectedModelId(m.id)}
                  className={clsx(
                    'p-3.5 rounded-lg border-2 text-left transition-all flex flex-col justify-between',
                    isSelected
                      ? 'border-primary-500 bg-primary-500/10'
                      : 'border-dark-700 bg-dark-900/50 hover:border-primary-500/40'
                  )}
                >
                  <div>
                    <div className="flex items-center justify-between gap-2 mb-1.5">
                      <span className={clsx(
                        'text-[11px] font-mono uppercase px-2 py-0.5 rounded font-semibold',
                        isDP
                          ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                          : isFederated
                          ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                          : 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                      )}>
                        {isDP ? 'DP-FedAvg (Privacy)' : isFederated ? 'Federated (FedAvg)' : 'Centralized'}
                      </span>
                      {isSelected && <CheckCircle className="w-4 h-4 text-primary-400" />}
                    </div>
                    <p className="font-semibold text-sm text-dark-50">{m.name}</p>
                    <p className="text-xs text-dark-400 font-mono mt-0.5">
                      {m.architecture} • v: {m.version}
                    </p>
                  </div>
                  <div className="mt-3 pt-2 border-t border-dark-700/60 grid grid-cols-3 gap-1 text-[11px] font-mono">
                    <div>
                      <span className="text-dark-400 block">Acc</span>
                      <span className="text-green-400 font-semibold">{acc}</span>
                    </div>
                    <div>
                      <span className="text-dark-400 block">F1</span>
                      <span className="text-primary-400 font-semibold">{f1}</span>
                    </div>
                    <div>
                      <span className="text-dark-400 block">Recall</span>
                      <span className="text-dark-200">{recall}</span>
                    </div>
                  </div>
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Adaptive Model Router Decision Card */}
      {compat && (
        <div className="p-5 bg-dark-800/60 border border-dark-700 rounded-xl space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <span className="text-xs uppercase tracking-wider text-primary-400 font-semibold">
                Phase 7 — Adaptive Model Router & Strategy Selection
              </span>
              <h3 className="text-lg font-bold text-dark-50 mt-0.5">
                Selected Strategy: {(selectedMode || compat.selected_mode).toUpperCase()} — {activeModelObj ? activeModelObj.name : compat.selected_model_name}
              </h3>
            </div>
            <div className="flex items-center gap-3">
              <div className="px-3 py-1.5 bg-dark-900 rounded-lg border border-dark-700 text-center">
                <p className="text-[11px] text-dark-400">Compatibility Score</p>
                <p className="text-lg font-bold text-primary-400">{compat.compatibility_score}%</p>
              </div>
              <div className="px-3 py-1.5 bg-dark-900 rounded-lg border border-dark-700 text-center">
                <p className="text-[11px] text-dark-400">Feature Coverage</p>
                <p className="text-lg font-bold text-green-400">{compat.feature_coverage_score}%</p>
              </div>
            </div>
          </div>

          {/* Available Modes Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
            {compat.available_modes.map((m) => {
              const isSelected = (selectedMode || compat.selected_mode) === m.mode
              return (
                <button
                  key={m.mode}
                  type="button"
                  disabled={!m.compatible}
                  onClick={() => setSelectedMode(m.mode)}
                  className={clsx(
                    'p-3.5 rounded-lg border-2 text-left transition-all flex flex-col justify-between',
                    isSelected
                      ? 'border-primary-500 bg-primary-500/10'
                      : m.compatible
                      ? 'border-dark-700 bg-dark-900/50 hover:border-primary-500/40'
                      : 'border-dark-800 bg-dark-900/20 opacity-50 cursor-not-allowed'
                  )}
                >
                  <div>
                    <div className="flex items-center justify-between gap-2 mb-1.5">
                      <span className="text-xs font-mono uppercase px-2 py-0.5 rounded bg-dark-800 text-primary-400">
                        {m.mode}
                      </span>
                      {m.recommended && (
                        <span className="text-[10px] px-2 py-0.5 rounded-full bg-green-500/20 text-green-400 font-semibold">
                          Recommended
                        </span>
                      )}
                    </div>
                    <p className="font-semibold text-sm text-dark-50">{m.name}</p>
                    <p className="text-xs text-dark-400 mt-1">{m.description}</p>
                  </div>
                  <div className="mt-3 pt-2 border-t border-dark-700/60 flex items-center justify-between text-xs">
                    <span className={m.compatible ? 'text-green-400' : 'text-red-400'}>
                      {m.compatible ? '✓ Compatible' : '✗ Insufficient Schema Overlap'}
                    </span>
                    {isSelected && <CheckCircle className="w-4 h-4 text-primary-400" />}
                  </div>
                </button>
              )
            })}
          </div>

          {/* Why this model was selected & Limitations */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 pt-2">
            <div className="p-3.5 bg-dark-900/60 border border-dark-700 rounded-lg">
              <h5 className="text-xs font-semibold uppercase tracking-wider text-green-400 mb-2 flex items-center gap-1.5">
                <CheckCircle className="w-4 h-4" />
                Why This Detection Mode Was Selected
              </h5>
              <ul className="text-xs text-dark-200 space-y-1.5 list-disc list-inside">
                {compat.routing_reasons.map((r, idx) => (
                  <li key={idx}>{r}</li>
                ))}
              </ul>
              <div className="mt-3 pt-2 border-t border-dark-800 flex flex-wrap gap-3 text-xs font-mono text-dark-300">
                <span>Supervised Weight: {(compat.scoring_weights.supervised_weight * 100).toFixed(0)}%</span>
                <span>Anomaly Weight: {(compat.scoring_weights.anomaly_weight * 100).toFixed(0)}%</span>
                <span>Rule Weight: {(compat.scoring_weights.rule_weight * 100).toFixed(0)}%</span>
              </div>
            </div>

            <div className="p-3.5 bg-dark-900/60 border border-dark-700 rounded-lg">
              <h5 className="text-xs font-semibold uppercase tracking-wider text-yellow-400 mb-2 flex items-center gap-1.5">
                <Info className="w-4 h-4" />
                Technical Honesty & Known Limitations
              </h5>
              {compat.limitations.length > 0 ? (
                <ul className="text-xs text-dark-300 space-y-1.5 list-disc list-inside">
                  {compat.limitations.map((lim, idx) => (
                    <li key={idx}>{lim}</li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-dark-300">
                  All 70 CIC-IDS2017 features are satisfied via direct mapping or exact mathematical derivation. No schema limitations apply.
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Preprocessing Execution Card */}
      {preprocessStatus === 'running' && (
        <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg">
          <div className="flex items-center gap-3">
            <Loader2 className="w-5 h-5 text-primary-400 animate-spin" />
            <span className="text-sm text-dark-200">{preprocessMessage}</span>
          </div>
        </div>
      )}

      {preprocessStatus === 'completed' && featureStats && (
        <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg space-y-3">
          <h4 className="text-sm font-semibold text-dark-50 flex items-center gap-2">
            <Hash className="w-4 h-4 text-primary-400" />
            Phase 6 & 10 — Feature Derivation & Preprocessing Readiness
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div className="p-3 bg-dark-900/60 rounded-lg">
              <p className="text-dark-400">Features Processed</p>
              <p className="text-lg font-bold font-mono text-dark-50">{featureStats.features_processed}</p>
            </div>
            <div className="p-3 bg-dark-900/60 rounded-lg">
              <p className="text-dark-400">Scaler Pipeline</p>
              <p className="font-mono text-dark-50 mt-1">{featureStats.scaler_type}</p>
            </div>
            <div className="p-3 bg-dark-900/60 rounded-lg">
              <p className="text-dark-400">Safely Derived Features</p>
              <p className="text-lg font-bold font-mono text-blue-400">{featureStats.derived_features_count ?? 0}</p>
            </div>
            <div className="p-3 bg-dark-900/60 rounded-lg">
              <p className="text-dark-400">Neutralized Baseline Features</p>
              <p className="text-lg font-bold font-mono text-yellow-400">{featureStats.neutralized_features_count ?? 0}</p>
            </div>
          </div>
        </div>
      )}

      {preprocessStatus === 'failed' && (
        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center justify-between">
          <div className="flex items-center gap-3">
            <AlertCircle className="w-6 h-6 text-red-400" />
            <div>
              <h4 className="font-semibold text-dark-50">Preprocessing Failed</h4>
              <p className="text-xs text-dark-300">{preprocessMessage}</p>
            </div>
          </div>
          <button
            onClick={() => { setPreprocessStatus('idle'); runPreprocessing() }}
            className="px-3 py-1.5 bg-primary-500 text-dark-950 rounded-lg text-xs font-medium"
          >
            Retry
          </button>
        </div>
      )}

      <div className="flex justify-between">
        <button onClick={onBack} className="px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg flex items-center gap-2 text-dark-300 hover:bg-dark-700">
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        <button
          onClick={onNext}
          disabled={preprocessStatus === 'failed'}
          className="px-5 py-2.5 bg-primary-500 text-dark-950 font-medium rounded-lg flex items-center gap-2 hover:bg-primary-400"
        >
          Execute 18-Stage Adaptive Detection Pipeline
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}

// Step 5: Real 18-Stage Traceable Pipeline Progress Component
function InferenceProgressStep({
  jobId,
  onComplete,
  onBack,
}: {
  jobId: string | null
  onComplete: () => void
  onBack: () => void
}) {
  const [status, setStatus] = useState<'running' | 'completed' | 'failed'>('running')
  const [message, setMessage] = useState('Executing 18-stage universal adaptive detection pipeline...')
  const [stages, setStages] = useState<StageTelemetry[]>([])
  const [summary, setSummary] = useState<Record<string, any> | null>(null)

  useEffect(() => {
    if (!jobId) return

    const pollForResults = async () => {
      try {
        const [jobRes, stagesRes] = await Promise.all([
          api.getInferenceJob(jobId),
          api.getInferenceStages(jobId),
        ])
        if (stagesRes.data?.stages) {
          setStages(stagesRes.data.stages)
        }
        if (stagesRes.data?.summary_metrics) {
          setSummary(stagesRes.data.summary_metrics)
        }

        if (jobRes.data.status === 'completed') {
          setStatus('completed')
          setMessage('All 18 pipeline stages completed with verified telemetry.')
        } else if (jobRes.data.status === 'failed') {
          setStatus('failed')
          setMessage(jobRes.data.error_message || 'Pipeline execution failed')
        }
      } catch (e) {
        console.error('Stage telemetry fetch failed:', e)
      }
    }

    pollForResults()
    const pollInterval = setInterval(pollForResults, 1000)
    return () => clearInterval(pollInterval)
  }, [jobId])

  const completedStagesCount = stages.filter(s => s.status === 'completed' || s.status === 'warning').length
  const totalTimeMs = stages.reduce((acc, s) => acc + (s.execution_time_ms || 0), 0)

  return (
    <div className="space-y-6 animate-in">
      <div className={clsx(
        'p-4 rounded-xl border flex flex-wrap items-center justify-between gap-4',
        status === 'completed' ? 'bg-green-500/10 border-green-500/30' :
        status === 'failed' ? 'bg-red-500/10 border-red-500/30' :
        'bg-dark-800/60 border-dark-700'
      )}>
        <div className="flex items-center gap-3">
          {status === 'completed' ? (
            <CheckCircle className="w-7 h-7 text-green-400" />
          ) : status === 'failed' ? (
            <AlertCircle className="w-7 h-7 text-red-400" />
          ) : (
            <Loader2 className="w-7 h-7 text-primary-400 animate-spin" />
          )}
          <div>
            <h3 className="text-lg font-bold text-dark-50">
              {status === 'completed' ? '18-Stage Adaptive Detection Pipeline Completed' :
               status === 'failed' ? 'Pipeline Execution Halted on Critical Error' :
               'Executing 18-Stage Adaptive Detection Pipeline...'}
            </h3>
            <p className="text-xs text-dark-300">{message}</p>
          </div>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="px-3 py-1.5 bg-dark-900 rounded-lg">
            <span className="text-dark-400">Stages: </span>
            <span className="text-dark-50 font-bold">{completedStagesCount} / 18</span>
          </div>
          <div className="px-3 py-1.5 bg-dark-900 rounded-lg">
            <span className="text-dark-400">Execution Time: </span>
            <span className="text-primary-400 font-bold">{totalTimeMs.toFixed(1)} ms</span>
          </div>
        </div>
      </div>

      {/* 18-Stage Telemetry Table */}
      {stages.length > 0 && (
        <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-xl">
          <h4 className="text-sm font-semibold text-dark-50 mb-3 flex items-center gap-2">
            <Activity className="w-4 h-4 text-primary-400" />
            Phase 8 — Real Stage-by-Stage Pipeline Execution Log
          </h4>
          <div className="overflow-x-auto max-h-[380px] border border-dark-700 rounded-lg">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-dark-900 border-b border-dark-700">
                <tr>
                  <th className="py-2 px-3 text-left text-dark-300">Stage</th>
                  <th className="py-2 px-3 text-left text-dark-300">Status</th>
                  <th className="py-2 px-3 text-left text-dark-300">Time (ms)</th>
                  <th className="py-2 px-3 text-left text-dark-300">Input → Output</th>
                  <th className="py-2 px-3 text-left text-dark-300">Telemetry Metadata & Warnings</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-800">
                {stages.map((st) => (
                  <tr key={st.stage_id} className="hover:bg-dark-800/40">
                    <td className="py-2 px-3 font-medium text-dark-50">{st.stage_name}</td>
                    <td className="py-2 px-3">
                      <span className={clsx('px-2 py-0.5 rounded-full text-[11px] font-medium uppercase', getStatusColor(st.status))}>
                        {st.status}
                      </span>
                    </td>
                    <td className="py-2 px-3 font-mono text-dark-300">{st.execution_time_ms.toFixed(2)} ms</td>
                    <td className="py-2 px-3 font-mono text-dark-300">
                      {st.input_count.toLocaleString()} → {st.output_count.toLocaleString()}
                    </td>
                    <td className="py-2 px-3 text-dark-300 font-mono text-[11px]">
                      {st.errors?.length > 0 && (
                        <span className="text-red-400 block">Error: {st.errors.join('; ')}</span>
                      )}
                      {st.warnings?.length > 0 && (
                        <span className="text-yellow-400 block">Warn: {st.warnings[0]}</span>
                      )}
                      {st.metadata && Object.keys(st.metadata).length > 0 && (
                        <span className="text-dark-400">
                          {Object.entries(st.metadata)
                            .slice(0, 4)
                            .map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : v}`)
                            .join(' • ')}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {summary && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="p-3.5 bg-dark-800/50 border border-dark-700 rounded-lg text-center">
            <p className="text-xs text-dark-400">Processed Records</p>
            <p className="text-2xl font-bold text-dark-50">{(summary.total_records ?? 0).toLocaleString()}</p>
          </div>
          <div className="p-3.5 bg-dark-800/50 border border-dark-700 rounded-lg text-center">
            <p className="text-xs text-dark-400">Normal Traffic</p>
            <p className="text-2xl font-bold text-green-400">{(summary.normal_count ?? 0).toLocaleString()}</p>
          </div>
          <div className="p-3.5 bg-dark-800/50 border border-dark-700 rounded-lg text-center">
            <p className="text-xs text-dark-400">Flagged / Suspicious</p>
            <p className="text-2xl font-bold text-red-400">{(summary.suspicious_or_malicious_count ?? 0).toLocaleString()}</p>
          </div>
          <div className="p-3.5 bg-dark-800/50 border border-dark-700 rounded-lg text-center">
            <p className="text-xs text-dark-400">Active Detection Mode</p>
            <p className="text-lg font-bold text-primary-400 uppercase">{summary.detection_mode || 'supervised'}</p>
          </div>
        </div>
      )}

      <div className="flex justify-between">
        <button onClick={onBack} className="px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg flex items-center gap-2 text-dark-300 hover:bg-dark-700">
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        {status === 'completed' && (
          <button onClick={onComplete} className="px-5 py-2.5 bg-primary-500 text-dark-950 font-medium rounded-lg flex items-center gap-2 hover:bg-primary-400">
            Explore Detection Analytics & Record Explanations
            <ArrowRight className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  )
}

// Step 6: Rich Detection Analytics & Record Investigation Component
function ResultsStep({
  job,
  results,
  isLoading,
  onExport,
  onViewDetections,
}: {
  job: InferenceJob | null
  results: DetectionResult[]
  isLoading: boolean
  onExport: () => void
  onViewDetections: () => void
}) {
  const [filterPrediction, setFilterPrediction] = useState<'all' | 'benign' | 'malicious'>('all')
  const [filterSeverity, setFilterSeverity] = useState<string>('all')
  const [resultsPage, setResultsPage] = useState(1)
  const [selectedRecord, setSelectedRecord] = useState<DetectionResult | null>(null)

  const summary = job?.config?.summary_metrics || {}
  const severityBreakdown: Record<string, number> = summary.severity_breakdown || {}
  const categoryBreakdown: Record<string, number> = summary.category_breakdown || {}
  const riskDist: Record<string, number> = summary.risk_score_distribution || {}
  const confDist: Record<string, number> = summary.confidence_distribution || {}
  const gtEval = summary.ground_truth_evaluation

  const filteredResults = results.filter((r) => {
    if (filterPrediction === 'benign' && r.prediction !== 0) return false
    if (filterPrediction === 'malicious' && r.prediction !== 1) return false
    if (filterSeverity !== 'all' && (r.severity || 'informational') !== filterSeverity) return false
    return true
  })

  const totalPages = Math.max(1, Math.ceil(filteredResults.length / 20))
  const paginatedResults = filteredResults.slice((resultsPage - 1) * 20, resultsPage * 20)

  const totalCount = job?.total_records ?? results.length
  const benignCount = job?.benign_count ?? results.filter(r => r.prediction === 0).length
  const maliciousCount = job?.malicious_count ?? results.filter(r => r.prediction === 1).length

  return (
    <div className="space-y-6 animate-in">
      {/* Top KPI Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="p-3.5 bg-dark-800/60 border border-dark-700 rounded-xl text-center">
          <p className="text-xs text-dark-400">Total Processed</p>
          <p className="text-2xl font-bold text-dark-50">{totalCount.toLocaleString()}</p>
          <p className="text-[11px] text-dark-400 mt-0.5">
            {job?.processing_time ? `${(job.processing_time * 1000).toFixed(0)} ms` : 'Completed'}
          </p>
        </div>
        <div className="p-3.5 bg-dark-800/60 border border-dark-700 rounded-xl text-center">
          <p className="text-xs text-dark-400">Normal Records</p>
          <p className="text-2xl font-bold text-green-400">{benignCount.toLocaleString()}</p>
          <p className="text-[11px] text-dark-400 mt-0.5">Baseline traffic</p>
        </div>
        <div className="p-3.5 bg-dark-800/60 border border-dark-700 rounded-xl text-center">
          <p className="text-xs text-dark-400">Confirmed Attacks</p>
          <p className="text-2xl font-bold text-red-400">
            {(summary.confirmed_attack_count ?? maliciousCount).toLocaleString()}
          </p>
          <p className="text-[11px] text-dark-400 mt-0.5">Supervised / Consensus</p>
        </div>
        <div className="p-3.5 bg-dark-800/60 border border-dark-700 rounded-xl text-center">
          <p className="text-xs text-dark-400">Probable Suspicious</p>
          <p className="text-2xl font-bold text-orange-400">
            {(summary.probable_suspicious_count ?? 0).toLocaleString()}
          </p>
          <p className="text-[11px] text-dark-400 mt-0.5">Hybrid signal</p>
        </div>
        <div className="p-3.5 bg-dark-800/60 border border-dark-700 rounded-xl text-center">
          <p className="text-xs text-dark-400">Statistical Anomalies</p>
          <p className="text-2xl font-bold text-yellow-400">
            {(summary.anomaly_detection_count ?? 0).toLocaleString()}
          </p>
          <p className="text-[11px] text-dark-400 mt-0.5">Outlier flows</p>
        </div>
        <div className="p-3.5 bg-dark-800/60 border border-dark-700 rounded-xl text-center">
          <p className="text-xs text-dark-400">Rule Alerts</p>
          <p className="text-2xl font-bold text-blue-400">
            {(summary.rule_alert_count ?? 0).toLocaleString()}
          </p>
          <p className="text-[11px] text-dark-400 mt-0.5">Heuristic evidence</p>
        </div>
      </div>

      {/* Ground-Truth Evaluation Banner (if dataset was labeled) */}
      {gtEval && (
        <div className="p-4 bg-primary-500/10 border border-primary-500/30 rounded-xl flex flex-wrap items-center justify-between gap-4">
          <div>
            <span className="text-xs uppercase font-semibold text-primary-400">
              Labeled Dataset Ground-Truth Evaluation ({gtEval.evaluated_records.toLocaleString()} rows)
            </span>
            <p className="text-xs text-dark-300 mt-0.5">
              Confusion Matrix: TP={gtEval.tp.toLocaleString()} • TN={gtEval.tn.toLocaleString()} • FP={gtEval.fp.toLocaleString()} • FN={gtEval.fn.toLocaleString()}
            </p>
          </div>
          <div className="flex flex-wrap gap-4 text-center font-mono text-sm">
            <div><span className="text-dark-400 text-xs block">Accuracy</span><span className="text-dark-50 font-bold">{(gtEval.accuracy * 100).toFixed(2)}%</span></div>
            <div><span className="text-dark-400 text-xs block">Precision</span><span className="text-dark-50 font-bold">{(gtEval.precision * 100).toFixed(2)}%</span></div>
            <div><span className="text-dark-400 text-xs block">Recall</span><span className="text-dark-50 font-bold">{(gtEval.recall * 100).toFixed(2)}%</span></div>
            <div><span className="text-dark-400 text-xs block">F1 Score</span><span className="text-primary-400 font-bold">{(gtEval.f1 * 100).toFixed(2)}%</span></div>
          </div>
        </div>
      )}

      {/* Severity, Category & Score Distributions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Severity Breakdown */}
        <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-xl">
          <h5 className="text-xs font-semibold uppercase tracking-wider text-dark-300 mb-3">
            Severity Breakdown
          </h5>
          <div className="space-y-2 text-xs">
            {(['critical', 'high', 'medium', 'low', 'informational'] as const).map((sev) => {
              const cnt = severityBreakdown[sev] ?? 0
              const pct = totalCount > 0 ? (cnt / totalCount) * 100 : 0
              return (
                <div key={sev} className="space-y-1">
                  <div className="flex justify-between">
                    <span className="capitalize text-dark-200 font-medium">{sev}</span>
                    <span className="font-mono text-dark-300">{cnt.toLocaleString()} ({pct.toFixed(1)}%)</span>
                  </div>
                  <div className="w-full h-1.5 bg-dark-900 rounded-full overflow-hidden">
                    <div
                      className={clsx(
                        'h-full rounded-full',
                        sev === 'critical' ? 'bg-red-500' :
                        sev === 'high' ? 'bg-orange-500' :
                        sev === 'medium' ? 'bg-yellow-500' :
                        sev === 'low' ? 'bg-blue-500' : 'bg-dark-500'
                      )}
                      style={{ width: `${Math.min(100, pct)}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Detection Category Breakdown */}
        <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-xl">
          <h5 className="text-xs font-semibold uppercase tracking-wider text-dark-300 mb-3">
            Detection Category & Mode Breakdown
          </h5>
          <div className="space-y-2 text-xs">
            {Object.entries(categoryBreakdown).map(([cat, cnt]) => (
              <div key={cat} className="flex items-center justify-between p-2 bg-dark-900/60 rounded-lg">
                <span className="text-dark-200 truncate max-w-[200px]" title={cat}>{cat}</span>
                <span className="font-mono font-semibold text-primary-400">{cnt.toLocaleString()}</span>
              </div>
            ))}
            <div className="pt-2 border-t border-dark-700 text-[11px] text-dark-400">
              Mode: <span className="text-dark-200 font-mono uppercase">{summary.detection_mode || 'supervised'}</span> •
              Model: <span className="text-dark-200">{summary.model_used || 'CICIDS2017_SUPERVISED_MODEL'}</span>
            </div>
          </div>
        </div>

        {/* Risk & Confidence Distributions */}
        <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-xl space-y-3">
          <h5 className="text-xs font-semibold uppercase tracking-wider text-dark-300">
            Risk Score & Confidence Distributions
          </h5>
          <div className="space-y-1.5 text-xs">
            <p className="text-[11px] text-dark-400">Risk Score Buckets</p>
            {Object.entries(riskDist).map(([bucket, cnt]) => {
              const pct = totalCount > 0 ? (cnt / totalCount) * 100 : 0
              return (
                <div key={bucket} className="flex items-center gap-2">
                  <span className="w-16 font-mono text-[11px] text-dark-400">{bucket}</span>
                  <div className="flex-1 h-1.5 bg-dark-900 rounded-full overflow-hidden">
                    <div className="h-full bg-primary-500 rounded-full" style={{ width: `${Math.min(100, pct)}%` }} />
                  </div>
                  <span className="w-12 text-right font-mono text-[11px] text-dark-300">{cnt}</span>
                </div>
              )
            })}
          </div>
          {Object.keys(confDist).length > 0 && (
            <div className="space-y-1 text-xs pt-2 border-t border-dark-700/60">
              <p className="text-[11px] text-dark-400">Confidence Buckets</p>
              {Object.entries(confDist).map(([bucket, cnt]) => {
                const pct = totalCount > 0 ? (cnt / totalCount) * 100 : 0
                return (
                  <div key={bucket} className="flex items-center gap-2">
                    <span className="w-16 font-mono text-[11px] text-dark-400">{bucket}</span>
                    <div className="flex-1 h-1 bg-dark-900 rounded-full overflow-hidden">
                      <div className="h-full bg-green-500 rounded-full" style={{ width: `${Math.min(100, pct)}%` }} />
                    </div>
                    <span className="w-12 text-right font-mono text-[11px] text-dark-300">{cnt}</span>
                  </div>
                )
              })}
            </div>
          )}
          <div className="pt-2 border-t border-dark-700 flex justify-between text-xs font-mono text-dark-300">
            <span>Mean Risk: {(summary.mean_risk_score ?? 0).toFixed(3)}</span>
            <span>Mean Confidence: {(summary.mean_confidence ?? 0).toFixed(3)}</span>
          </div>
        </div>
      </div>

      {/* Filters & Export Bar */}
      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-lg flex flex-wrap items-center gap-3">
        <div className="flex gap-1.5">
          {(['all', 'benign', 'malicious'] as const).map((filter) => (
            <button
              key={filter}
              onClick={() => { setFilterPrediction(filter); setResultsPage(1) }}
              className={clsx(
                'px-3 py-1.5 text-xs font-medium rounded-lg transition-colors',
                filterPrediction === filter
                  ? 'bg-primary-500 text-dark-950'
                  : 'bg-dark-900 text-dark-300 hover:bg-dark-700'
              )}
            >
              {filter === 'all' ? 'All Records' : filter === 'benign' ? 'Normal Only' : 'Flagged / Suspicious'}
            </button>
          ))}
        </div>

        <select
          value={filterSeverity}
          onChange={(e) => { setFilterSeverity(e.target.value); setResultsPage(1) }}
          className="px-3 py-1.5 bg-dark-900 border border-dark-700 rounded-lg text-xs text-dark-200"
        >
          <option value="all">All Severities</option>
          <option value="critical">Critical</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
          <option value="informational">Informational</option>
        </select>

        <div className="ml-auto flex items-center gap-2">
          <button onClick={onExport} className="px-3.5 py-1.5 bg-green-500/15 text-green-400 rounded-lg hover:bg-green-500/25 flex items-center gap-1.5 text-xs font-medium">
            <Download className="w-4 h-4" />
            Export Enriched CSV
          </button>
          <button onClick={onViewDetections} className="px-3.5 py-1.5 bg-primary-500/15 text-primary-400 rounded-lg hover:bg-primary-500/25 flex items-center gap-1.5 text-xs font-medium">
            <BarChart2 className="w-4 h-4" />
            Open in Detections Explorer
          </button>
        </div>
      </div>

      {/* Records Table */}
      <div className="p-4 bg-dark-800/50 border border-dark-700 rounded-xl">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <Loader2 className="w-8 h-8 animate-spin text-primary-400" />
          </div>
        ) : filteredResults.length === 0 ? (
          <div className="flex items-center justify-center h-40 text-dark-400 text-sm">
            No records match the current filter.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="bg-dark-900 border-b border-dark-700">
                <tr>
                  <th className="py-2.5 px-3 text-left text-dark-300">#</th>
                  <th className="py-2.5 px-3 text-left text-dark-300">Predicted Category</th>
                  <th className="py-2.5 px-3 text-left text-dark-300">Severity</th>
                  <th className="py-2.5 px-3 text-left text-dark-300">Mode</th>
                  <th className="py-2.5 px-3 text-left text-dark-300">Risk Score</th>
                  <th className="py-2.5 px-3 text-left text-dark-300">Supervised / Anomaly / Rule</th>
                  <th className="py-2.5 px-3 text-left text-dark-300">Confidence</th>
                  <th className="py-2.5 px-3 text-left text-dark-300">Explanation</th>
                  <th className="py-2.5 px-3 text-left text-dark-300">Inspect</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-800">
                {paginatedResults.map((result) => {
                  const risk = result.risk_score ?? result.probability
                  return (
                    <tr
                      key={result.id}
                      onClick={() => setSelectedRecord(result)}
                      className="hover:bg-dark-800/60 cursor-pointer transition-colors"
                    >
                      <td className="py-2.5 px-3 font-mono text-dark-400">{result.record_index}</td>
                      <td className="py-2.5 px-3">
                        <span className={clsx(
                          'px-2 py-0.5 rounded-full text-[11px] font-medium',
                          result.prediction === 1 ? 'bg-red-500/20 text-red-300' : 'bg-green-500/20 text-green-400'
                        )}>
                          {result.predicted_category || (result.prediction === 1 ? 'Flagged' : 'Normal Traffic')}
                        </span>
                      </td>
                      <td className="py-2.5 px-3">
                        <span className={clsx('px-2 py-0.5 rounded text-[10px] font-mono uppercase', getSeverityBadge(result.severity))}>
                          {result.severity || 'informational'}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 font-mono uppercase text-dark-300">
                        {result.detection_mode || 'supervised'}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-dark-200">
                        {risk.toFixed(4)}
                        <div className="w-20 h-1 bg-dark-700 rounded-full mt-1 overflow-hidden">
                          <div
                            className={clsx('h-full rounded-full', result.prediction === 1 ? 'bg-red-500' : 'bg-green-500')}
                            style={{ width: `${(risk * 100).toFixed(0)}%` }}
                          />
                        </div>
                      </td>
                      <td className="py-2.5 px-3 font-mono text-dark-400">
                        {result.supervised_probability !== null && result.supervised_probability !== undefined ? result.supervised_probability.toFixed(3) : '—'}
                        {' / '}
                        {result.anomaly_score !== null && result.anomaly_score !== undefined ? result.anomaly_score.toFixed(3) : '—'}
                        {' / '}
                        {(result.rule_score ?? 0).toFixed(2)}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-dark-300">{(result.confidence ?? 0).toFixed(3)}</td>
                      <td className="py-2.5 px-3 text-dark-300 max-w-[260px] truncate" title={result.explanation}>
                        {result.explanation || '-'}
                      </td>
                      <td className="py-2.5 px-3">
                        <button
                          onClick={(e) => { e.stopPropagation(); setSelectedRecord(result) }}
                          className="p-1.5 rounded bg-dark-900 text-primary-400 hover:bg-dark-700"
                        >
                          <Eye className="w-3.5 h-3.5" />
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}

        {totalPages > 1 && (
          <div className="mt-4 flex items-center justify-center gap-2 text-xs">
            <button
              onClick={() => setResultsPage(Math.max(1, resultsPage - 1))}
              disabled={resultsPage === 1}
              className="px-2.5 py-1 bg-dark-900 border border-dark-700 rounded text-dark-300 disabled:opacity-50"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="text-dark-400 px-3">Page {resultsPage} of {totalPages}</span>
            <button
              onClick={() => setResultsPage(Math.min(totalPages, resultsPage + 1))}
              disabled={resultsPage === totalPages}
              className="px-2.5 py-1 bg-dark-900 border border-dark-700 rounded text-dark-300 disabled:opacity-50"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>

      {/* Record Investigation Modal */}
      {selectedRecord && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in">
          <div className="w-full max-w-3xl max-h-[90vh] bg-dark-900 border border-dark-700 rounded-xl overflow-hidden flex flex-col">
            <div className="p-4 border-b border-dark-700 flex items-center justify-between">
              <div>
                <span className="text-xs uppercase font-semibold text-primary-400">
                  Phase 10 — Flow Record Investigation
                </span>
                <h3 className="text-lg font-bold text-dark-50">
                  Record #{selectedRecord.record_index} — {selectedRecord.predicted_category || (selectedRecord.prediction === 1 ? 'Flagged' : 'Normal')}
                </h3>
              </div>
              <button
                onClick={() => setSelectedRecord(null)}
                className="p-2 rounded-lg text-dark-400 hover:bg-dark-800"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-5 overflow-y-auto space-y-4 text-xs">
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                <div className="p-3 bg-dark-800 rounded-lg">
                  <p className="text-dark-400">Risk Score</p>
                  <p className="text-lg font-bold font-mono text-dark-50">{(selectedRecord.risk_score ?? selectedRecord.probability).toFixed(4)}</p>
                </div>
                <div className="p-3 bg-dark-800 rounded-lg">
                  <p className="text-dark-400">Supervised Prob</p>
                  <p className="text-lg font-bold font-mono text-primary-400">
                    {selectedRecord.supervised_probability !== null && selectedRecord.supervised_probability !== undefined
                      ? selectedRecord.supervised_probability.toFixed(4)
                      : 'N/A'}
                  </p>
                </div>
                <div className="p-3 bg-dark-800 rounded-lg">
                  <p className="text-dark-400">Anomaly Score</p>
                  <p className="text-lg font-bold font-mono text-yellow-400">
                    {selectedRecord.anomaly_score !== null && selectedRecord.anomaly_score !== undefined
                      ? selectedRecord.anomaly_score.toFixed(4)
                      : 'N/A'}
                  </p>
                </div>
                <div className="p-3 bg-dark-800 rounded-lg">
                  <p className="text-dark-400">Rule Score</p>
                  <p className="text-lg font-bold font-mono text-blue-400">{(selectedRecord.rule_score ?? 0).toFixed(4)}</p>
                </div>
                <div className="p-3 bg-dark-800 rounded-lg">
                  <p className="text-dark-400">Severity / Mode</p>
                  <p className="font-mono uppercase text-dark-50 mt-1">
                    {selectedRecord.severity || 'info'} • {selectedRecord.detection_mode || 'supervised'}
                  </p>
                </div>
              </div>

              <div className="p-3.5 bg-dark-800/70 border border-dark-700 rounded-lg">
                <p className="font-semibold text-dark-200 mb-1">Detection Explanation</p>
                <p className="text-dark-300 leading-relaxed">{selectedRecord.explanation}</p>
              </div>

              {selectedRecord.triggered_rules && selectedRecord.triggered_rules.length > 0 && (
                <div className="p-3.5 bg-dark-800/70 border border-dark-700 rounded-lg">
                  <p className="font-semibold text-dark-200 mb-2">Triggered Network Rules</p>
                  <div className="space-y-1.5">
                    {selectedRecord.triggered_rules.map((r, i) => (
                      <div key={i} className="flex items-center justify-between p-2 bg-dark-900 rounded">
                        <span className="font-medium text-orange-400">{r.name}</span>
                        <span className="font-mono text-dark-300">{r.evidence}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {selectedRecord.top_anomaly_features && selectedRecord.top_anomaly_features.length > 0 && (
                <div className="p-3.5 bg-dark-800/70 border border-dark-700 rounded-lg">
                  <p className="font-semibold text-dark-200 mb-1">Top Deviating Statistical Features</p>
                  <p className="font-mono text-yellow-400">{selectedRecord.top_anomaly_features.join(' • ')}</p>
                </div>
              )}

              {selectedRecord.features && Object.keys(selectedRecord.features).length > 0 && (
                <div className="p-3.5 bg-dark-800/70 border border-dark-700 rounded-lg">
                  <p className="font-semibold text-dark-200 mb-2">
                    Original Flow Feature Values ({Object.keys(selectedRecord.features).length} captured fields)
                  </p>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 max-h-60 overflow-y-auto">
                    {Object.entries(selectedRecord.features).map(([k, v]) => (
                      <div key={k} className="p-2 bg-dark-900 rounded flex justify-between gap-2">
                        <span className="text-dark-400 truncate" title={k}>{k}</span>
                        <span className="font-mono text-dark-100">{typeof v === 'number' ? Number(v).toFixed(2) : String(v)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Main Inference Component
export function Inference() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { data: jobs, isLoading: jobsLoading } = useInferenceJobs()

  const [currentStep, setCurrentStep] = useState(1)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [jobName, setJobName] = useState('')
  const [useSampleData, setUseSampleData] = useState<string | null>(null)
  const [selectedSample, setSelectedSample] = useState<typeof SAMPLE_DATASETS[0]>(SAMPLE_DATASETS[0])
  const [analysis, setAnalysis] = useState<DatasetAnalysis | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [validation, setValidation] = useState<ValidationReport | null>(null)
  const [mappingData, setMappingData] = useState<FeatureMappingPayload | null>(null)
  const [isValidating, setIsValidating] = useState(false)
  const [compatibleModels, setCompatibleModels] = useState<CompatibleModel[]>([])
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null)
  const [selectedMode, setSelectedMode] = useState<string>('')
  const [inferenceJobId, setInferenceJobId] = useState<string | null>(null)
  const [jobResults, setJobResults] = useState<DetectionResult[]>([])
  const [isLoadingResults, setIsLoadingResults] = useState(false)
  const [completedJob, setCompletedJob] = useState<InferenceJob | null>(null)

  const handleDeleteJob = async (jobId: string) => {
    if (!confirm('Delete this inference job and all its results?')) return
    try {
      await api.deleteInferenceJob(jobId)
      queryClient.invalidateQueries({ queryKey: ['inferenceJobs'] })
      queryClient.invalidateQueries({ queryKey: ['dashboardStats'] })
    } catch (error: any) {
      alert(`Failed to delete: ${error.response?.data?.detail || error.message}`)
    }
  }

  const handleSampleSelect = (sample: typeof SAMPLE_DATASETS[0]) => {
    setSelectedSample(sample)
    setUseSampleData(sample.id)
    setSelectedFile(null)
    setJobName(sample.name)
  }

  // Step 1 -> 2: Analyze & Profile Dataset
  const handleAnalyze = async () => {
    setIsAnalyzing(true)
    setValidation(null)
    setMappingData(null)
    setSelectedMode('')
    setInferenceJobId(null)
    setCompletedJob(null)
    setJobResults([])
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

  // Step 2 -> 3: Validate Dataset & Load Semantic Feature Mapping
  const handleValidate = async () => {
    if (!analysis) return
    setIsValidating(true)
    try {
      const [valRes, mapRes] = await Promise.all([
        api.validateDataset(analysis.dataset_id),
        api.getFeatureMapping(analysis.dataset_id),
      ])
      setValidation(valRes.data)
      setMappingData(mapRes.data)
      setSelectedMode(mapRes.data?.compatibility?.selected_mode || '')
      setCurrentStep(3)
    } catch (error: any) {
      console.error('Validation failed:', error)
      alert(`Validation failed: ${error.response?.data?.detail || error.message}`)
    } finally {
      setIsValidating(false)
    }
  }

  const handleSaveManualOverrides = async (overrides: Record<string, string>) => {
    if (!analysis) return
    try {
      const res = await api.updateFeatureMapping(analysis.dataset_id, overrides)
      setMappingData(res.data)
      if (res.data.validation) {
        setValidation(res.data.validation)
      }
      if (res.data.compatibility?.selected_mode) {
        setSelectedMode(res.data.compatibility.selected_mode)
      }
    } catch (error: any) {
      alert(`Failed to update feature mapping: ${error.response?.data?.detail || error.message}`)
    }
  }

  // Step 3 -> 4: Load Compatible Models & Proceed to Model Routing / Preprocessing
  const handleLoadModels = async () => {
    if (!analysis) return
    try {
      const data = await api.listCompatibleModels(analysis.dataset_id)
      const models: CompatibleModel[] = data.data || []
      setCompatibleModels(models)
      if (models.length > 0) {
        if (!selectedModelId || !models.some((m) => m.id === selectedModelId)) {
          // Prefer Federated models (FedAvg / DP-FedAvg) by default in a Federated IDS
          const fedModels = models.filter((m) => m.algorithm.toLowerCase().includes('fed'))
          const pool = fedModels.length > 0 ? fedModels : models
          const defaultModel = pool.reduce((best: CompatibleModel, current: CompatibleModel) => {
            const bestAcc = best.metrics?.accuracy ?? 0
            const currentAcc = current.metrics?.accuracy ?? 0
            return currentAcc > bestAcc ? current : best
          }, pool[0])
          setSelectedModelId(defaultModel.id)
        }
      } else {
        setSelectedModelId('a1f2e3d4-5678-49ab-9cde-fed200000020')
      }
      setCurrentStep(4)
    } catch {
      setSelectedModelId('a1f2e3d4-5678-49ab-9cde-fed200000020')
      setCurrentStep(4)
    }
  }

  // Step 4 -> 5: Execute 18-Stage Adaptive Inference Pipeline
  const handleRunInference = async () => {
    if (!analysis) return
    let activeModelId = selectedModelId || 'a1f2e3d4-5678-49ab-9cde-fed200000020'
    setInferenceJobId(null)
    setCurrentStep(5)

    try {
      const data = await api.runInferenceOnDataset({
        dataset_id: analysis.dataset_id,
        model_id: activeModelId,
        name: jobName || analysis.filename,
        threshold: 0.5,
        batch_size: 512,
        detection_mode: selectedMode || undefined,
      })
      setInferenceJobId(data.data.id)
      queryClient.invalidateQueries({ queryKey: ['inferenceJobs'] })
      queryClient.invalidateQueries({ queryKey: ['dashboardStats'] })
    } catch (error: any) {
      console.error('Inference failed:', error)
      alert(`Inference failed: ${error.response?.data?.detail || error.message}`)
      setCurrentStep(4)
    }
  }

  // Step 5 -> 6: Load Enriched Detection Results
  const handleLoadResults = async (jobIdOverride?: string) => {
    const targetJobId = typeof jobIdOverride === 'string' ? jobIdOverride : inferenceJobId
    if (!targetJobId) return
    setIsLoadingResults(true)
    try {
      const [jobRes, resultsRes] = await Promise.all([
        api.getInferenceJob(targetJobId),
        api.getInferenceResults(targetJobId, { limit: 2000 }),
      ])
      setCompletedJob(jobRes.data)
      const items = Array.isArray(resultsRes.data) ? resultsRes.data : (resultsRes.data?.items || [])
      setJobResults(items)
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
    const headers = [
      'Record Index', 'Predicted Category', 'Prediction', 'Severity', 'Detection Mode',
      'Risk Score', 'Supervised Probability', 'Anomaly Score', 'Rule Score', 'Confidence', 'True Label', 'Explanation'
    ]
    const rows = jobResults.map(r => [
      r.record_index,
      `"${r.predicted_category || (r.prediction === 1 ? 'Flagged' : 'Normal')}"`,
      r.prediction === 1 ? 'Flagged' : 'Normal',
      r.severity || 'informational',
      r.detection_mode || 'supervised',
      (r.risk_score ?? r.probability).toFixed(4),
      r.supervised_probability !== null && r.supervised_probability !== undefined ? r.supervised_probability.toFixed(4) : '',
      r.anomaly_score !== null && r.anomaly_score !== undefined ? r.anomaly_score.toFixed(4) : '',
      (r.rule_score ?? 0).toFixed(4),
      (r.confidence ?? 0).toFixed(4),
      r.true_label !== undefined && r.true_label !== null ? (r.true_label === 1 ? 'Attack' : 'Benign') : 'N/A',
      `"${(r.explanation || '').replace(/"/g, '""')}"`
    ])
    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `adaptive_detections_${completedJob.name}_${Date.now()}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleRestart = () => {
    setCurrentStep(1)
    setSelectedFile(null)
    setJobName('')
    setUseSampleData(null)
    setAnalysis(null)
    setValidation(null)
    setMappingData(null)
    setSelectedModelId(null)
    setSelectedMode('')
    setInferenceJobId(null)
    setJobResults([])
    setCompletedJob(null)
  }

  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-50">Universal Dataset Intelligence & Adaptive Detection</h1>
          <p className="text-dark-400 mt-1">
            Schema-aware ingestion, semantic feature mapping, safe feature derivation, and multi-mode intrusion detection
          </p>
        </div>
        {currentStep > 1 && (
          <button
            onClick={handleRestart}
            className="px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg flex items-center gap-2 text-dark-300 hover:bg-dark-700"
          >
            <RefreshCw className="w-4 h-4" />
            New Dataset Run
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
                  isActive ? 'bg-primary-500/15 border border-primary-500/40' :
                  isCompleted ? 'bg-green-500/10 border border-green-500/30' :
                  'bg-dark-900/50 border border-dark-700'
                )}>
                  {isCompleted ? (
                    <CheckCircle className="w-5 h-5 text-green-400" />
                  ) : (
                    <step.icon className={clsx('w-5 h-5', isActive ? 'text-primary-400' : 'text-dark-400')} />
                  )}
                  <span className={clsx('font-medium text-sm', isActive ? 'text-primary-400' : isCompleted ? 'text-green-400' : 'text-dark-400')}>
                    {step.name}
                  </span>
                </div>
                {index < WORKFLOW_STEPS.length - 1 && (
                  <div className={clsx('w-6 h-0.5 mx-1', isCompleted ? 'bg-green-500' : 'bg-dark-700')} />
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Step Content */}
      <div className="p-6 bg-dark-900/50 border border-dark-700 rounded-xl min-h-[480px]">
        {currentStep === 1 && (
          <UploadStep
            onNext={handleAnalyze}
            onSampleSelect={handleSampleSelect}
            selectedFile={selectedFile}
            setSelectedFile={(f) => { setSelectedFile(f); if (f) setUseSampleData(null) }}
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
            mappingData={mappingData}
            isLoading={isValidating}
            onSaveOverrides={handleSaveManualOverrides}
            onNext={handleLoadModels}
            onBack={() => setCurrentStep(2)}
          />
        )}

        {currentStep === 4 && (
          <PreprocessingStep
            analysis={analysis}
            mappingData={mappingData}
            compatibleModels={compatibleModels}
            selectedModelId={selectedModelId}
            setSelectedModelId={setSelectedModelId}
            selectedMode={selectedMode}
            setSelectedMode={setSelectedMode}
            onNext={handleRunInference}
            onBack={() => setCurrentStep(3)}
          />
        )}

        {currentStep === 5 && (
          <InferenceProgressStep
            jobId={inferenceJobId}
            onComplete={() => handleLoadResults()}
            onBack={() => setCurrentStep(4)}
          />
        )}

        {currentStep === 6 && (
          <ResultsStep
            job={completedJob}
            results={jobResults}
            isLoading={isLoadingResults}
            onExport={handleExport}
            onViewDetections={() => navigate('/detections')}
          />
        )}
      </div>

      {/* Recent Jobs Table */}
      <div className="p-6 rounded-xl border border-dark-700 bg-dark-900/50">
        <h3 className="text-lg font-semibold text-dark-50 mb-4">Recent Adaptive Detection Jobs</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-dark-700">
                <th className="pb-3 font-medium text-dark-300">Name</th>
                <th className="pb-3 font-medium text-dark-300">Model</th>
                <th className="pb-3 font-medium text-dark-300">Status</th>
                <th className="pb-3 font-medium text-dark-300">Mode / Schema</th>
                <th className="pb-3 font-medium text-dark-300">Records</th>
                <th className="pb-3 font-medium text-dark-300">Normal</th>
                <th className="pb-3 font-medium text-dark-300">Flagged</th>
                <th className="pb-3 font-medium text-dark-300">Created</th>
                <th className="pb-3 font-medium text-dark-300">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-800">
              {jobsLoading ? (
                <tr>
                  <td className="py-4 text-dark-400 text-center" colSpan={9}>
                    <Loader2 className="w-5 h-5 animate-spin inline mr-2" />
                    Loading...
                  </td>
                </tr>
              ) : jobs && jobs.length > 0 ? (
                jobs.map((job: InferenceJob) => (
                  <tr key={job.id} className="hover:bg-dark-800/50 transition-colors">
                    <td className="py-3 font-medium text-dark-50">{job.name}</td>
                    <td className="py-3 text-xs font-mono text-emerald-300">
                      {job.model_name || job.config?.summary_metrics?.model_used || 'Federated v2'}
                    </td>
                    <td className="py-3">
                      <span className={clsx('px-2 py-0.5 text-xs rounded-full', getStatusColor(job.status))}>
                        {job.status}
                      </span>
                    </td>
                    <td className="py-3 font-mono text-xs text-primary-400">
                      {(job.config?.detection_mode || 'supervised').toUpperCase()}
                      {job.config?.schema_family ? ` (${job.config.schema_family})` : ''}
                    </td>
                    <td className="py-3 text-dark-300">{job.total_records?.toLocaleString() ?? '-'}</td>
                    <td className="py-3 text-green-400">{job.benign_count?.toLocaleString() ?? '-'}</td>
                    <td className="py-3 text-red-400">{job.malicious_count?.toLocaleString() ?? '-'}</td>
                    <td className="py-3 text-dark-400 text-xs">
                      {job.created_at ? format(new Date(job.created_at), 'MMM d, HH:mm') : '-'}
                    </td>
                    <td className="py-3">
                      <div className="flex items-center gap-2">
                        {job.status === 'completed' && (
                          <button
                            onClick={async () => {
                              setInferenceJobId(job.id)
                              await handleLoadResults(job.id)
                            }}
                            className="px-3 py-1 text-xs bg-primary-500/15 text-primary-400 rounded-lg hover:bg-primary-500/25 flex items-center gap-1"
                          >
                            <FileText className="w-3.5 h-3.5" />
                            View Analytics
                          </button>
                        )}
                        <button
                          onClick={() => handleDeleteJob(job.id)}
                          className="p-1.5 text-xs bg-red-500/10 text-red-400 rounded-lg hover:bg-red-500/20"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td className="py-8 text-dark-400 text-center" colSpan={8}>
                    No detection jobs yet. Upload or select a dataset above to run the adaptive pipeline.
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
