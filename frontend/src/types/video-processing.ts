/**
 * Types related to video processing
 */

export interface Detection {
  frame_number: number;
  class_name: string;
  confidence: number;
  bbox: number[];
  service: string;
}

export interface ProcessingResult {
  processing_id: string;
  status: string;
  video_info?: {
    total_frames: number;
    fps: number;
    duration_seconds: number;
    sampled_frames: number;
  };
  processing_time?: {
    yolo: number;
    aws: number;
    azure: number;
  };
  detections?: {
    yolo: Detection[];
    aws: Detection[];
    azure: Detection[];
  };
  total_detections?: {
    yolo: number;
    aws: number;
    azure: number;
  };
  costs?: {
    yolo: number;
    aws: number;
    azure: number;
  };
  error?: string;
}

export interface ProviderMetrics {
  avg_processing_time: number;
  total_videos_processed: number;
  accuracy: {
    precision: number;
    recall: number;
    f1_score?: number;
  };
  avg_cost: number;
  total_cost: number;
  detection_counts: {
    people: number;
    vehicles: number;
    other: number;
  };
}

export interface MetricsResponse {
  yolo: ProviderMetrics;
  aws: ProviderMetrics;
  azure: ProviderMetrics;
  time_period: string;
} 