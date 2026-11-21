import { useQuery } from '@tanstack/react-query';
import { api } from '@/services/api';

export function useDashboardStats() {
  return useQuery({
    queryKey: ['dashboardStats'],
    queryFn: async () => {
      const response = await api.getStats();
      return response.data;
    },
    refetchInterval: 30000,
  });
}

export function useOrganisations() {
  return useQuery({
    queryKey: ['organisations'],
    queryFn: async () => {
      const response = await api.getOrganisations({ page_size: 100 });
      return response.data.items;
    },
    refetchInterval: 30000,
  });
}

export function useExperiments() {
  return useQuery({
    queryKey: ['experiments'],
    queryFn: async () => {
      const response = await api.getExperiments({ page_size: 100 });
      return response.data.items;
    },
    refetchInterval: 30000,
  });
}

export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: async () => {
      const response = await api.getModels({ page_size: 100 });
      return response.data.items;
    },
    refetchInterval: 30000,
  });
}

export function useInferenceJobs() {
  return useQuery({
    queryKey: ['inferenceJobs'],
    queryFn: async () => {
      const response = await api.getInferenceJobs({ page_size: 100 });
      return response.data.items;
    },
    refetchInterval: 10000,
  });
}

export function useDatasets() {
  return useQuery({
    queryKey: ['datasets'],
    queryFn: async () => {
      const response = await api.getDatasets({ page_size: 100 });
      return response.data.items;
    },
    refetchInterval: 30000,
  });
}