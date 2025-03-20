import api from './api';
import { Model, ModelMetrics, CloudMetric, CloudPerformanceMetric } from '../types/model';

interface ModelRegistrationData {
    name: string;
    version: string;
    type: 'yolo' | 'rcnn' | 'faster-rcnn';
    framework: string;
    cloudPlatform: 'aws' | 'azure' | 'gcp';
    endpointUrl: string;
    accuracy: number;
}

export const getModels = async (params?: { status?: string; platform?: string }): Promise<Model[]> => {
    const response = await api.get<Model[]>('/models/list', { params });
    return response.data;
};

export const registerModel = async (data: ModelRegistrationData): Promise<Model> => {
    const response = await api.post<Model>('/models/register', data);
    return response.data;
};

export const deployModel = async (modelId: number): Promise<void> => {
    await api.post(`/models/${modelId}/deploy`);
};

export const getModelMetrics = async (modelId: number, timeRange: '24h' | '7d' | '30d' = '7d'): Promise<ModelMetrics[]> => {
    const response = await api.get<ModelMetrics[]>(`/models/${modelId}/metrics`, {
        params: { range: timeRange }
    });
    return response.data;
};

export const compareModels = async (): Promise<Record<string, { avgProcessingTime: number; totalRequests: number }>> => {
    const response = await api.get('/models/compare');
    return response.data;
};

export const getCloudCosts = async (): Promise<Record<string, CloudMetric[]>> => {
    const response = await api.get('/cloud/costs');
    return response.data;
};

export const getCloudPerformance = async (): Promise<CloudPerformanceMetric[]> => {
    const response = await api.get('/cloud/performance');
    return response.data;
};

export const analyzeImage = async (imageUrl: string, cloudPlatform: 'aws' | 'azure' | 'gcp') => {
    const response = await api.post('/detections/analyze', {
        imageUrl,
        cloudPlatform
    });
    return response.data;
}; 