import React, { useState, useRef, useEffect } from 'react';
import { Button, Card, Typography, Space, Alert, Table, Tabs, Spin, Progress, Statistic } from 'antd';
import { UploadOutlined, LineChartOutlined, CheckCircleOutlined, ClockCircleOutlined } from '@ant-design/icons';
import { uploadVideo, getProcessingStatus } from '../../services/videoProcessing';
import { ProcessingResult } from '../../types/video-processing';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;

// Time interval for polling status updates (in ms)
const POLLING_INTERVAL = 5000;

const ComparisonPanel: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [processingId, setProcessingId] = useState<string | null>(null);
  const [result, setResult] = useState<ProcessingResult | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollingTimerRef = useRef<NodeJS.Timeout | null>(null);

  const isFileSystemAccessSupported = 'showOpenFilePicker' in window;

  // Start polling when we have a processing ID
  useEffect(() => {
    if (processingId && !result?.status?.includes('completed') && !result?.status?.includes('failed')) {
      setIsPolling(true);
      pollStatus();
    } else {
      setIsPolling(false);
    }

    return () => {
      if (pollingTimerRef.current) {
        clearTimeout(pollingTimerRef.current);
      }
    };
  }, [processingId, result]);

  const pollStatus = async () => {
    if (!processingId) return;
    
    try {
      const statusResult = await getProcessingStatus(processingId);
      setResult(statusResult);
      
      // Continue polling if not complete
      if (!statusResult.status?.includes('completed') && !statusResult.status?.includes('failed')) {
        pollingTimerRef.current = setTimeout(pollStatus, POLLING_INTERVAL);
      } else {
        setIsPolling(false);
      }
    } catch (err) {
      console.error('Error polling status:', err);
      setIsPolling(false);
    }
  };

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

  const startProcessing = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setError(null);

    try {
      const uploadResult = await uploadVideo(selectedFile);
      setProcessingId(uploadResult.processing_id);
      setResult(uploadResult);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
      setError(errorMessage);
    } finally {
      setIsUploading(false);
    }
  };

  const getStatusDisplay = () => {
    if (!result) return null;
    
    if (result.status === 'processing') {
      return (
        <div style={{ textAlign: 'center', margin: '20px 0' }}>
          <Spin size="large" />
          <Paragraph style={{ marginTop: 16 }}>
            Processing your video. This may take several minutes depending on the file size.
          </Paragraph>
          <Progress 
            percent={30} 
            status="active" 
            strokeColor={{ from: '#108ee9', to: '#87d068' }}
          />
        </div>
      );
    } else if (result.status === 'completed') {
      return (
        <Alert
          message="Processing Complete"
          description="Your video has been successfully processed by all providers."
          type="success"
          showIcon
          icon={<CheckCircleOutlined />}
        />
      );
    } else if (result.status === 'failed') {
      return (
        <Alert
          message="Processing Failed"
          description={result.error || "An unknown error occurred during processing."}
          type="error"
          showIcon
        />
      );
    }
    
    return (
      <Alert
        message="Status"
        description={`Current status: ${result.status}`}
        type="info"
        showIcon
        icon={<ClockCircleOutlined />}
      />
    );
  };

  const renderResults = () => {
    if (!result || !result.processing_time) return null;
    
    return (
      <Tabs defaultActiveKey="summary">
        <TabPane tab="Summary" key="summary">
          <Card title="Processing Time (seconds)">
            <Space size="large">
              <Statistic title="YOLO (Local)" value={result.processing_time.yolo.toFixed(2)} suffix="s" />
              <Statistic title="AWS Rekognition" value={result.processing_time.aws.toFixed(2)} suffix="s" />
              <Statistic title="Azure AI Vision" value={result.processing_time.azure.toFixed(2)} suffix="s" />
            </Space>
          </Card>

          {result.total_detections && (
            <Card title="Objects Detected" style={{ marginTop: 16 }}>
              <Space size="large">
                <Statistic title="YOLO (Local)" value={result.total_detections.yolo} />
                <Statistic title="AWS Rekognition" value={result.total_detections.aws} />
                <Statistic title="Azure AI Vision" value={result.total_detections.azure} />
              </Space>
            </Card>
          )}

          {result.costs && (
            <Card title="Estimated Cost (USD)" style={{ marginTop: 16 }}>
              <Space size="large">
                <Statistic title="YOLO (Local)" value={result.costs.yolo.toFixed(4)} prefix="$" />
                <Statistic title="AWS Rekognition" value={result.costs.aws.toFixed(4)} prefix="$" />
                <Statistic title="Azure AI Vision" value={result.costs.azure.toFixed(4)} prefix="$" />
              </Space>
            </Card>
          )}
        </TabPane>

        <TabPane tab="Video Info" key="video">
          {result.video_info && (
            <Card>
              <Paragraph><strong>Total Frames:</strong> {result.video_info.total_frames}</Paragraph>
              <Paragraph><strong>FPS:</strong> {result.video_info.fps}</Paragraph>
              <Paragraph><strong>Duration:</strong> {result.video_info.duration_seconds.toFixed(2)} seconds</Paragraph>
              <Paragraph><strong>Sampled Frames:</strong> {result.video_info.sampled_frames}</Paragraph>
            </Card>
          )}
        </TabPane>

        <TabPane tab="Detailed Comparison" key="details">
          <Paragraph>
            This tab would display detailed comparison results, detection accuracy metrics, 
            and performance analysis between the three detection providers.
          </Paragraph>
        </TabPane>
      </Tabs>
    );
  };

  return (
    <Card title="Cloud Provider Comparison" style={{ maxWidth: 900, margin: '20px auto' }}>
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
              disabled={isUploading || isPolling}
            >
              Select Video
            </Button>
            {selectedFile && <Text type="secondary">{selectedFile.name}</Text>}
          </Space>
        </div>

        {/* Start Processing */}
        <div>
          <Title level={4}>2. Process Video</Title>
          <Button 
            type="primary"
            onClick={startProcessing}
            disabled={!selectedFile || isUploading || isPolling}
            loading={isUploading}
          >
            {isUploading ? 'Uploading...' : 'Start Processing'}
          </Button>
          <Paragraph type="secondary" style={{ marginTop: 8 }}>
            Video will be processed by AWS Rekognition, Azure AI Vision, and local YOLO model
          </Paragraph>
        </div>

        {/* Status Display */}
        {(processingId || error) && (
          <div>
            <Title level={4}>3. Processing Status</Title>
            {error ? (
              <Alert
                message="Error"
                description={error}
                type="error"
                showIcon
              />
            ) : getStatusDisplay()}
          </div>
        )}

        {/* Results Display */}
        {result && result.status === 'completed' && (
          <div>
            <Title level={4}>4. Results</Title>
            {renderResults()}
          </div>
        )}
      </Space>
    </Card>
  );
};

export default ComparisonPanel; 