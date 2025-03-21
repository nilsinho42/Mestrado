import React, { useState, useRef } from 'react';
import { Button, Card, Typography, Space, Alert } from 'antd';
import { UploadOutlined, LineChartOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

interface Detection {
  frame_number: number;
  class: string;
  confidence: number;
  bbox: number[];
  service: string;
}

interface ComparisonResult {
  video_info: {
    total_frames: number;
    fps: number;
    duration_seconds: number;
    sampled_frames: number;
    frame_indices: number[];
  };
  processing_time: {
    yolo: number;
    aws: number;
    azure: number;
  };
  detections: {
    yolo: Detection[];
    aws: Detection[];
    azure: Detection[];
  };
  total_detections: {
    yolo: number;
    aws: number;
    azure: number;
  };
  frames: {
    total: number;
    sampled: number;
    processed: {
      yolo: number;
      aws: number;
      azure: number;
    }
  };
  costs: {
    yolo: number;
    aws: number;
    azure: number;
  };
  dashboard_url: string;
  saved_frames: {
    yolo: string[];
    aws: string[];
    azure: string[];
  };
}

const ComparisonPanel: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ComparisonResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const isFileSystemAccessSupported = 'showOpenFilePicker' in window;

  const handleFileSelect = async () => {
    try {
      if (isFileSystemAccessSupported) {
        const [fileHandle] = await window.showOpenFilePicker({
          types: [
            {
              description: 'Video Files',
              accept: {
                'video/*': ['.mp4', '.avi', '.mov']
              }
            }
          ]
        });
        
        const file = await fileHandle.getFile();
        setSelectedFile(file);
        setError(null);
      } else {
        // Fallback to traditional file input
        fileInputRef.current?.click();
      }
    } catch (err) {
      if (err instanceof Error && err.name !== 'AbortError') {
        setError('Failed to select file');
      }
    }
  };

  const handleFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
    }
  };

  const startComparison = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('video', selectedFile);

      const response = await fetch('http://localhost:8000/start_comparison', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start comparison');
      }

      const data = await response.json() as ComparisonResult;
      setResult(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
      setError(errorMessage);
    } finally {
      setIsProcessing(false);
    }
  };

  const openDashboard = () => {
    if (result?.dashboard_url) {
      window.open(result.dashboard_url, '_blank', 'noopener,noreferrer');
    }
  };

  return (
    <Card title="Service Comparison" style={{ maxWidth: 800, margin: '0 auto' }}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* Hidden file input for fallback */}
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept="video/*,.mp4,.avi,.mov"
          onChange={handleFileInputChange}
        />

        {/* File Selection */}
        <div>
          <Title level={4}>1. Select Video File</Title>
          <Space>
            <Button 
              icon={<UploadOutlined />} 
              onClick={handleFileSelect}
              disabled={isProcessing}
            >
              Select Video
            </Button>
            {selectedFile && <Text type="secondary">{selectedFile.name}</Text>}
          </Space>
        </div>

        {/* Start Comparison */}
        <div>
          <Title level={4}>2. Start Comparison</Title>
          <Button 
            type="primary"
            onClick={startComparison}
            disabled={!selectedFile || isProcessing}
            loading={isProcessing}
          >
            Start Comparison
          </Button>
        </div>

        {/* Error Display */}
        {error && (
          <Alert
            message="Error"
            description={error}
            type="error"
            showIcon
          />
        )}

        {/* Results Display */}
        {result && (
          <div>
            <Title level={4}>3. Results</Title>
            <Space direction="vertical">
              <Card size="small" title="Video Information">
                <Text>Total Frames: {result.video_info.total_frames}</Text>
                <br />
                <Text>FPS: {result.video_info.fps}</Text>
                <br />
                <Text>Duration: {result.video_info.duration_seconds.toFixed(2)} seconds</Text>
                <br />
                <Text>Sampled Frames: {result.video_info.sampled_frames}</Text>
              </Card>

              <Card size="small" title="Frames Processed">
                <Text>YOLO: {result.frames.processed.yolo} / {result.frames.sampled}</Text>
                <br />
                <Text>AWS: {result.frames.processed.aws} / {result.frames.sampled}</Text>
                <br />
                <Text>Azure: {result.frames.processed.azure} / {result.frames.sampled}</Text>
              </Card>

              <Card size="small" title="Processing Time (seconds)">
                <Text>YOLO: {result.processing_time.yolo.toFixed(3)}</Text>
                <br />
                <Text>AWS: {result.processing_time.aws.toFixed(3)}</Text>
                <br />
                <Text>Azure: {result.processing_time.azure.toFixed(3)}</Text>
              </Card>

              <Card size="small" title="Total Detections">
                <Text>YOLO: {result.total_detections.yolo}</Text>
                <br />
                <Text>AWS: {result.total_detections.aws}</Text>
                <br />
                <Text>Azure: {result.total_detections.azure}</Text>
              </Card>

              <Card size="small" title="Estimated Cost (USD)">
                <Text>YOLO: ${result.costs.yolo.toFixed(6)}</Text>
                <br />
                <Text>AWS: ${result.costs.aws.toFixed(6)}</Text>
                <br />
                <Text>Azure: ${result.costs.azure.toFixed(6)}</Text>
              </Card>

              <Button 
                type="primary" 
                icon={<LineChartOutlined />}
                onClick={openDashboard}
              >
                Open Detailed Dashboard
              </Button>
            </Space>
          </div>
        )}
      </Space>
    </Card>
  );
};

export default ComparisonPanel; 