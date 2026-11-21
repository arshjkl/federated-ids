import axios, { AxiosInstance, AxiosError } from 'axios';

const API_BASE_URL = (import.meta as unknown as { env: { VITE_API_URL?: string; VITE_WS_URL?: string } }).env.VITE_API_URL || '';
const WS_BASE_URL = (import.meta as unknown as { env: { VITE_API_URL?: string; VITE_WS_URL?: string } }).env.VITE_WS_URL || (
  API_BASE_URL
    ? API_BASE_URL.replace(/^http:/, 'ws:').replace(/^https:/, 'wss:')
    : (typeof window !== 'undefined' && window.location.host ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}` : 'ws://localhost:18003')
);

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 120000,
    });

    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        const requestUrl = error.config?.url ? `${error.config.baseURL || ''}${error.config.url}` : 'unknown';
        const method = error.config?.method?.toUpperCase() || 'UNKNOWN';

        if (error.response) {
          console.error(`API Error [${method} ${requestUrl}]:`, {
            status: error.response.status,
            statusText: error.response.statusText,
            data: error.response.data,
            headers: error.response.headers,
          });
        } else if (error.request) {
          console.error(`API Network Error [${method} ${requestUrl}]:`, {
            message: error.message,
            code: error.code,
            errno: (error as any).errno,
            syscall: (error as any).syscall,
            address: (error as any).address,
            port: (error as any).port,
            timeout: error.config?.timeout,
          });
        } else {
          console.error(`API Request Setup Error [${method} ${requestUrl}]:`, {
            message: error.message,
            config: error.config,
          });
        }
        return Promise.reject(error);
      }
    );
  }

  // Health
  async healthCheck() {
    return this.client.get('/health');
  }

  // Stats
  async getStats() {
    const response = await this.client.get('/api/stats');
    if (!response.data) {
      throw new Error('Empty response from stats endpoint');
    }
    return response;
  }

  // Organisations
  async getOrganisations(params?: { page?: number; page_size?: number; status?: string }) {
    return this.client.get('/api/organisations', { params });
  }

  async getOrganisation(id: string) {
    return this.client.get(`/api/organisations/${id}`);
  }

  async createOrganisation(data: any) {
    return this.client.post('/api/organisations', data);
  }

  async updateOrganisation(id: string, data: any) {
    return this.client.patch(`/api/organisations/${id}`, data);
  }

  async deleteOrganisation(id: string) {
    return this.client.delete(`/api/organisations/${id}`);
  }

  // Experiments
  async getExperiments(params?: { page?: number; page_size?: number; status?: string }) {
    return this.client.get('/api/experiments', { params });
  }

  async getExperiment(id: string) {
    return this.client.get(`/api/experiments/${id}`);
  }

  async createExperiment(data: any) {
    return this.client.post('/api/experiments', data);
  }

  async updateExperiment(id: string, data: any) {
    return this.client.patch(`/api/experiments/${id}`, data);
  }

  async deleteExperiment(id: string) {
    return this.client.delete(`/api/experiments/${id}`);
  }

  async startExperiment(id: string) {
    return this.client.post(`/api/experiments/${id}/start`);
  }

  async pauseExperiment(id: string) {
    return this.client.post(`/api/experiments/${id}/pause`);
  }

  async resumeExperiment(id: string) {
    return this.client.post(`/api/experiments/${id}/resume`);
  }

  async stopExperiment(id: string) {
    return this.client.post(`/api/experiments/${id}/stop`);
  }

  // Models
  async getModels(params?: { page?: number; page_size?: number; status?: string; algorithm?: string }) {
    return this.client.get('/api/models', { params });
  }

  async getModel(id: string) {
    return this.client.get(`/api/models/${id}`);
  }

  async createModel(data: any) {
    return this.client.post('/api/models', data);
  }

  async updateModel(id: string, data: any) {
    return this.client.patch(`/api/models/${id}`, data);
  }

  async deleteModel(id: string) {
    return this.client.delete(`/api/models/${id}`);
  }

  async deleteInferenceJob(id: string) {
    return this.client.delete(`/api/inference/${id}`);
  }

  // Inference
  async getInferenceJobs(params?: { page?: number; page_size?: number; status?: string }) {
    return this.client.get('/api/inference', { params });
  }

  async getInferenceJob(id: string) {
    return this.client.get(`/api/inference/${id}`);
  }

  async createInferenceJob(data: any) {
    return this.client.post('/api/inference', data);
  }

  async getInferenceResults(id: string, params?: { limit?: number; offset?: number; prediction?: number }) {
    return this.client.get(`/api/inference/${id}/results`, { params });
  }

  async uploadCSVAndRunInference(formData: FormData) {
    return this.client.post('/api/inference/upload-csv', formData);
  }

  async uploadSampleAndRunInference(data: { model_id?: string; organisation_id?: string; name: string; sample_file: string; threshold?: number; batch_size?: number; detection_mode?: string; manual_overrides?: Record<string, string> }) {
    const formData = new FormData();
    if (data.model_id) formData.append('model_id', data.model_id);
    if (data.organisation_id) formData.append('organisation_id', data.organisation_id);
    formData.append('name', data.name);
    formData.append('sample_file', data.sample_file);
    if (data.threshold !== undefined) formData.append('threshold', String(data.threshold));
    if (data.batch_size !== undefined) formData.append('batch_size', String(data.batch_size));
    if (data.detection_mode) formData.append('detection_mode', data.detection_mode);
    if (data.manual_overrides) formData.append('manual_overrides', JSON.stringify(data.manual_overrides));
    return this.client.post('/api/inference/upload-sample', formData);
  }

  async runInferenceOnDataset(data: { dataset_id: string; model_id?: string; organisation_id?: string; name: string; threshold?: number; batch_size?: number; detection_mode?: string; manual_overrides?: Record<string, string> }) {
    const formData = new FormData();
    formData.append('dataset_id', data.dataset_id);
    if (data.model_id) formData.append('model_id', data.model_id);
    if (data.organisation_id) formData.append('organisation_id', data.organisation_id);
    formData.append('name', data.name);
    if (data.threshold !== undefined) formData.append('threshold', String(data.threshold));
    if (data.batch_size !== undefined) formData.append('batch_size', String(data.batch_size));
    if (data.detection_mode) formData.append('detection_mode', data.detection_mode);
    if (data.manual_overrides) formData.append('manual_overrides', JSON.stringify(data.manual_overrides));
    return this.client.post('/api/inference/run-on-dataset', formData);
  }

  // Dataset Analysis & Universal Intelligence
  async analyzeCSVDataset(formData: FormData) {
    return this.client.post('/api/inference/analyze-csv', formData);
  }

  async analyzeSampleDataset(sampleFile: string) {
    const formData = new FormData();
    formData.append('sample_file', sampleFile);
    return this.client.post('/api/inference/analyze-sample', formData);
  }

  async previewDataset(datasetId: string, params?: { page?: number; page_size?: number; search?: string; sort_by?: string; sort_order?: string }) {
    return this.client.get(`/api/inference/preview/${datasetId}`, { params });
  }

  async getDatasetProfile(datasetId: string) {
    return this.client.get(`/api/inference/profile/${datasetId}`);
  }

  async validateDataset(datasetId: string) {
    return this.client.get(`/api/inference/validate/${datasetId}`);
  }

  async getFeatureMapping(datasetId: string) {
    return this.client.get(`/api/inference/feature-mapping/${datasetId}`);
  }

  async updateFeatureMapping(datasetId: string, manualOverrides: Record<string, string>) {
    return this.client.post(`/api/inference/feature-mapping/${datasetId}`, {
      manual_overrides: manualOverrides,
    });
  }

  async getDatasetCompatibility(datasetId: string, requestedMode?: string) {
    return this.client.get(`/api/inference/compatibility/${datasetId}`, {
      params: requestedMode ? { requested_mode: requestedMode } : undefined,
    });
  }

  async preprocessDataset(datasetId: string) {
    return this.client.post(`/api/inference/preprocess/${datasetId}`);
  }

  async getInferenceStages(jobId: string) {
    return this.client.get(`/api/inference/${jobId}/stages`);
  }

  async getRecordInvestigation(jobId: string, recordIndex: number) {
    return this.client.get(`/api/inference/${jobId}/records/${recordIndex}`);
  }

  async listCompatibleModels(datasetId?: string) {
    return this.client.get('/api/models/compatible', { params: { dataset_id: datasetId } });
  }

  // Detections
  async getDetectionResults(jobId: string, params?: {
    limit?: number;
    offset?: number;
    prediction?: number;
    min_probability?: number;
    max_probability?: number;
    start_date?: string;
    end_date?: string;
  }) {
    return this.client.get(`/api/inference/${jobId}/results`, { params });
  }

  async getAllDetections(params?: {
    page?: number;
    page_size?: number;
    search?: string;
    model_id?: string;
    dataset_id?: string;
    organisation_id?: string;
    prediction?: number;
    min_probability?: number;
    max_probability?: number;
    start_date?: string;
    end_date?: string;
  }) {
    return this.client.get('/api/detections', { params });
  }

  async exportDetectionsCSV(jobId: string, params?: {
    prediction?: number;
    include_features?: boolean;
  }) {
    return this.client.get(`/api/inference/${jobId}/export`, {
      params,
      responseType: 'blob'
    });
  }

  // Datasets
  async getDatasets(params?: { page?: number; page_size?: number; status?: string }) {
    return this.client.get('/api/datasets', { params });
  }

  async createDataset(data: any) {
    return this.client.post('/api/datasets', data);
  }

  // WebSocket
  createWebSocket(path: string): WebSocket {
    return new WebSocket(`${WS_BASE_URL}${path}`);
  }
}

export const api = new ApiService();
export default api;
