import axios, { AxiosInstance, AxiosError } from 'axios';

const API_BASE_URL = (import.meta as unknown as { env: { VITE_API_URL?: string; VITE_WS_URL?: string } }).env.VITE_API_URL || 'http://localhost:8000';
const WS_BASE_URL = (import.meta as unknown as { env: { VITE_API_URL?: string; VITE_WS_URL?: string } }).env.VITE_WS_URL || 'ws://localhost:8000';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    });

    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        console.error('API Error:', error.response?.status, error.response?.data);
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
    return this.client.get('/api/stats');
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
    return this.client.post('/api/inference/upload-csv', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
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