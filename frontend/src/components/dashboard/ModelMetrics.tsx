import React, { useEffect, useState } from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer
} from 'recharts';
import { getModelMetrics } from '../../services/models';
import { ModelMetrics as ModelMetricsType } from '../../types/model';

interface Props {
    modelId: number;
    timeRange: '24h' | '7d' | '30d';
}

export const ModelMetrics: React.FC<Props> = ({ modelId, timeRange }) => {
    const [metrics, setMetrics] = useState<ModelMetricsType[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                setLoading(true);
                const data = await getModelMetrics(modelId, timeRange);
                setMetrics(data || []);
                setError(null);
            } catch (err) {
                setError('Failed to load model metrics');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        fetchMetrics();
    }, [modelId, timeRange]);

    if (loading) return <div>Loading metrics...</div>;
    if (error) return <div className="error">{error}</div>;
    if (!metrics || !metrics.length) return <div>No metrics available</div>;

    return (
        <div className="model-metrics">
            <h3>Model Performance Metrics</h3>
            
            {/* Inference Time Chart */}
            <div className="chart-container">
                <h4>Average Inference Time</h4>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={metrics}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                            dataKey="date" 
                            tickFormatter={(date) => new Date(date).toLocaleDateString()}
                        />
                        <YAxis />
                        <Tooltip 
                            labelFormatter={(date) => new Date(date).toLocaleDateString()}
                            formatter={(value: number) => `${value.toFixed(3)}ms`}
                        />
                        <Legend />
                        <Line 
                            type="monotone" 
                            dataKey="avgInferenceTime" 
                            stroke="#8884d8" 
                            name="Avg. Inference Time"
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            {/* Confidence Score Chart */}
            <div className="chart-container">
                <h4>Average Confidence Score</h4>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={metrics}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                            dataKey="date" 
                            tickFormatter={(date) => new Date(date).toLocaleDateString()}
                        />
                        <YAxis domain={[0, 1]} />
                        <Tooltip 
                            labelFormatter={(date) => new Date(date).toLocaleDateString()}
                            formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                        />
                        <Legend />
                        <Line 
                            type="monotone" 
                            dataKey="avgConfidence" 
                            stroke="#82ca9d" 
                            name="Avg. Confidence"
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            {/* Request Volume Chart */}
            <div className="chart-container">
                <h4>Daily Inference Count</h4>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={metrics}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                            dataKey="date" 
                            tickFormatter={(date) => new Date(date).toLocaleDateString()}
                        />
                        <YAxis />
                        <Tooltip 
                            labelFormatter={(date) => new Date(date).toLocaleDateString()}
                        />
                        <Legend />
                        <Line 
                            type="monotone" 
                            dataKey="inferenceCount" 
                            stroke="#ffc658" 
                            name="Inference Count"
                        />
                        <Line 
                            type="monotone" 
                            dataKey="errorCount" 
                            stroke="#ff7300" 
                            name="Error Count"
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}; 