import api from './api';
import { ProcessingResult, MetricsResponse } from '../types/video-processing';

/**
 * Uploads a video file and starts processing
 */
export const uploadVideo = async (file: File): Promise<ProcessingResult> => {
  const formData = new FormData();
  formData.append('video', file);
  
  const response = await api.post('/api/videos/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

/**
 * Gets the status of video processing
 */
export const getProcessingStatus = async (processingId: string): Promise<ProcessingResult> => {
  const response = await api.get(`/api/videos/${processingId}/status`);
  return response.data;
};

/**
 * Gets metrics for all providers
 */
export const getMetrics = async (): Promise<MetricsResponse> => {
  const response = await api.get('/api/metrics');
  return response.data;
};

/**
 * Gets metrics for a specific provider (aws, azure, or yolo)
 */
export const getProviderMetrics = async (provider: string): Promise<any> => {
  const response = await api.get(`/api/metrics/${provider}`);
  return response.data;
}; 