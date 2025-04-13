import api from './api';
import { ProcessingResult, MetricsResponse } from '../types/video-processing';

/**
 * Uploads a video file and starts processing
 */
export const uploadVideo = async (file: File): Promise<ProcessingResult> => {
  console.log('Starting video upload:', file.name, 'Size:', file.size);
  
  // Create a simple FormData with just the video file
  const formData = new FormData();
  formData.append('video', file);
  
  console.log('FormData created, sending request to:', '/api/videos/upload');
  
  try {
    // Use more robust configuration for large file uploads
    const response = await api.post('/api/videos/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      // Add timeout to prevent hanging indefinitely
      timeout: 300000, // 5 minutes
      // Disable default transformRequest to ensure FormData is sent correctly
      transformRequest: [(data) => data],
    });
    
    console.log('Upload response received:', response.data);
    return response.data;
  } catch (error: any) {
    console.error('Upload error:', error.message);
    if (error.response) {
      console.error('Error response:', error.response.status, error.response.data);
    } else if (error.request) {
      console.error('No response received, request:', error.request);
    }
    throw error;
  }
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