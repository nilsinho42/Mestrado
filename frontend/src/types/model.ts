export interface Model {
    id: number;
    name: string;
    version: string;
    type: 'yolo' | 'rcnn' | 'faster-rcnn';
    framework: string;
    status: 'inactive' | 'deployed';
    cloudPlatform: 'aws' | 'azure' | 'gcp';
    endpointUrl: string;
    accuracy: number;
    avgInferenceTime: number;
    createdAt: string;
    updatedAt: string;
    deployedAt?: string;
    createdBy: number;
}

export interface ModelMetrics {
    id: number;
    modelId: number;
    date: string;
    inferenceCount: number;
    avgInferenceTime: number;
    avgConfidence: number;
    errorCount: number;
}

export interface Detection {
    id: number;
    imageId: number;
    class: string;
    score: number;
    boundingBox: {
        x: number;
        y: number;
        width: number;
        height: number;
    };
    createdAt: string;
}

export interface CloudMetric {
    date: string;
    requestCount: number;
    avgLatency: number;
    cost: number;
}

export interface CloudPerformanceMetric {
    platform: string;
    avgLatency: number;
    totalRequests: number;
    totalCost: number;
} 