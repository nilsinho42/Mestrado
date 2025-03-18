import React, { useState, useEffect } from 'react';
import { ModelMetrics } from './ModelMetrics';
import CloudComparison from './CloudComparison';
import { getModels } from '../../services/models';
import { Model } from '../../types/model';
import { useWebSocket } from '../../hooks/useWebSocket';

export const Dashboard: React.FC = () => {
    const [selectedModel, setSelectedModel] = useState<number | null>(null);
    const [timeRange, setTimeRange] = useState<'24h' | '7d' | '30d'>('24h');
    const [models, setModels] = useState<Model[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const { isConnected, lastMessage } = useWebSocket(
        `${import.meta.env.VITE_WS_URL || 'ws://localhost:8080'}/ws`
    );

    useEffect(() => {
        const fetchModels = async () => {
            try {
                setLoading(true);
                const data = await getModels({ status: 'deployed' });
                setModels(data);
                if (data.length > 0 && !selectedModel) {
                    setSelectedModel(data[0].id);
                }
                setError(null);
            } catch (err) {
                setError('Failed to load models');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        fetchModels();
    }, []);

    // Handle real-time updates
    useEffect(() => {
        if (lastMessage?.type === 'modelUpdate') {
            // Update metrics in real-time
            // This would trigger a re-render of the charts
            console.log('Real-time update received:', lastMessage.data);
        }
    }, [lastMessage]);

    return (
        <div className="dashboard">
            <header className="dashboard-header">
                <h2>Monitoring Dashboard</h2>
                <div className="connection-status">
                    Status: {isConnected ? 
                        <span className="connected">Connected</span> : 
                        <span className="disconnected">Disconnected</span>
                    }
                </div>
            </header>

            <div className="dashboard-controls">
                <div className="model-selector">
                    <label htmlFor="model">Model:</label>
                    <select
                        id="model"
                        value={selectedModel || ''}
                        onChange={(e) => setSelectedModel(Number(e.target.value))}
                    >
                        <option value="">Select a model</option>
                        {models.map((model) => (
                            <option key={model.id} value={model.id}>
                                {model.name} (v{model.version}) - {model.cloudPlatform}
                            </option>
                        ))}
                    </select>
                </div>

                <div className="time-range-selector">
                    <label htmlFor="timeRange">Time Range:</label>
                    <select
                        id="timeRange"
                        value={timeRange}
                        onChange={(e) => setTimeRange(e.target.value as '24h' | '7d' | '30d')}
                    >
                        <option value="24h">Last 24 Hours</option>
                        <option value="7d">Last 7 Days</option>
                        <option value="30d">Last 30 Days</option>
                    </select>
                </div>
            </div>

            <div className="dashboard-content">
                {loading ? (
                    <div className="loading">Loading dashboard data...</div>
                ) : error ? (
                    <div className="error">{error}</div>
                ) : (
                    <>
                        {selectedModel && (
                            <div className="metrics-section">
                                <ModelMetrics 
                                    modelId={selectedModel} 
                                    timeRange={timeRange} 
                                />
                            </div>
                        )}

                        <div className="comparison-section">
                            <CloudComparison />
                        </div>
                    </>
                )}
            </div>

            <style>{`
                .dashboard {
                    padding: 20px;
                    max-width: 1400px;
                    margin: 0 auto;
                }

                .dashboard-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }

                .connection-status {
                    font-size: 0.9em;
                }

                .connected {
                    color: #4caf50;
                }

                .disconnected {
                    color: #f44336;
                }

                .dashboard-controls {
                    display: flex;
                    gap: 20px;
                    margin-bottom: 20px;
                }

                .model-selector,
                .time-range-selector {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }

                select {
                    padding: 8px;
                    border-radius: 4px;
                    border: 1px solid #ccc;
                }

                .dashboard-content {
                    display: grid;
                    gap: 20px;
                }

                .metrics-section,
                .comparison-section {
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }

                .chart-container {
                    margin-bottom: 30px;
                }

                .loading,
                .error {
                    text-align: center;
                    padding: 20px;
                }

                .error {
                    color: #f44336;
                }

                @media (max-width: 768px) {
                    .dashboard-controls {
                        flex-direction: column;
                    }
                }
            `}</style>
        </div>
    );
}; 