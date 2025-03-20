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
import { Model, ModelMetrics as ModelMetricsType } from '../../types/model';

interface Props {
    models: Model[];
    timeRange: '24h' | '7d' | '30d';
}

// Array of colors for different model lines
const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#ff0000'];

export const ModelComparison: React.FC<Props> = ({ models, timeRange }) => {
    const [metricsMap, setMetricsMap] = useState<{ [key: number]: ModelMetricsType[] }>({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchAllMetrics = async () => {
            try {
                setLoading(true);
                const promises = models.map(model => 
                    getModelMetrics(model.id, timeRange)
                        .then(data => ({ modelId: model.id, data: data || [] }))
                );
                
                const results = await Promise.all(promises);
                const newMetricsMap = results.reduce((acc, { modelId, data }) => {
                    acc[modelId] = data;
                    return acc;
                }, {} as { [key: number]: ModelMetricsType[] });
                
                setMetricsMap(newMetricsMap);
                setError(null);
            } catch (err) {
                setError('Failed to load model metrics');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        fetchAllMetrics();
    }, [models, timeRange]);

    if (loading) return <div>Loading metrics...</div>;
    if (error) return <div className="error">{error}</div>;
    if (!Object.keys(metricsMap).length) return <div>No metrics available</div>;

    // Combine all dates from all models
    const allDates = new Set<string>();
    Object.values(metricsMap).forEach(metrics => {
        metrics.forEach(metric => {
            allDates.add(new Date(metric.date).toISOString());
        });
    });

    // Create combined data for charts
    const combinedData = Array.from(allDates).sort().map(date => {
        const dataPoint: any = { date };
        models.forEach(model => {
            const modelMetrics = metricsMap[model.id];
            const metric = modelMetrics?.find(m => 
                new Date(m.date).toISOString() === date
            );
            if (metric) {
                dataPoint[`inferenceTime_${model.id}`] = metric.avgInferenceTime;
                dataPoint[`confidence_${model.id}`] = metric.avgConfidence;
                dataPoint[`errors_${model.id}`] = metric.errorCount;
            }
        });
        return dataPoint;
    });

    return (
        <div className="model-comparison">
            <h3>Model Performance Comparison</h3>
            
            {/* Inference Time Chart */}
            <div className="chart-container">
                <h4>Average Inference Time Comparison</h4>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={combinedData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                            dataKey="date" 
                            tickFormatter={(date) => new Date(date).toLocaleDateString()}
                        />
                        <YAxis />
                        <Tooltip 
                            labelFormatter={(date) => new Date(date).toLocaleDateString()}
                            formatter={(value: number, name: string) => {
                                const modelId = parseInt(name.split('_')[1]);
                                const model = models.find(m => m.id === modelId);
                                return [`${value.toFixed(3)}ms`, model?.name || 'Unknown'];
                            }}
                        />
                        <Legend 
                            formatter={(value: string) => {
                                const modelId = parseInt(value.split('_')[1]);
                                const model = models.find(m => m.id === modelId);
                                return `${model?.name} (v${model?.version})`;
                            }}
                        />
                        {models.map((model, index) => (
                            <Line
                                key={model.id}
                                type="monotone"
                                dataKey={`inferenceTime_${model.id}`}
                                stroke={COLORS[index % COLORS.length]}
                                name={`inferenceTime_${model.id}`}
                                strokeWidth={2}
                                dot={false}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </div>

            {/* Confidence Score Chart */}
            <div className="chart-container">
                <h4>Average Confidence Score Comparison</h4>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={combinedData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                            dataKey="date" 
                            tickFormatter={(date) => new Date(date).toLocaleDateString()}
                        />
                        <YAxis />
                        <Tooltip 
                            labelFormatter={(date) => new Date(date).toLocaleDateString()}
                            formatter={(value: number, name: string) => {
                                const modelId = parseInt(name.split('_')[1]);
                                const model = models.find(m => m.id === modelId);
                                return [`${(value * 100).toFixed(1)}%`, model?.name || 'Unknown'];
                            }}
                        />
                        <Legend 
                            formatter={(value: string) => {
                                const modelId = parseInt(value.split('_')[1]);
                                const model = models.find(m => m.id === modelId);
                                return `${model?.name} (v${model?.version})`;
                            }}
                        />
                        {models.map((model, index) => (
                            <Line
                                key={model.id}
                                type="monotone"
                                dataKey={`confidence_${model.id}`}
                                stroke={COLORS[index % COLORS.length]}
                                name={`confidence_${model.id}`}
                                strokeWidth={2}
                                dot={false}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </div>

            {/* Error Count Chart */}
            <div className="chart-container">
                <h4>Error Count Comparison</h4>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={combinedData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                            dataKey="date" 
                            tickFormatter={(date) => new Date(date).toLocaleDateString()}
                        />
                        <YAxis />
                        <Tooltip 
                            labelFormatter={(date) => new Date(date).toLocaleDateString()}
                            formatter={(value: number, name: string) => {
                                const modelId = parseInt(name.split('_')[1]);
                                const model = models.find(m => m.id === modelId);
                                return [value, model?.name || 'Unknown'];
                            }}
                        />
                        <Legend 
                            formatter={(value: string) => {
                                const modelId = parseInt(value.split('_')[1]);
                                const model = models.find(m => m.id === modelId);
                                return `${model?.name} (v${model?.version})`;
                            }}
                        />
                        {models.map((model, index) => (
                            <Line
                                key={model.id}
                                type="monotone"
                                dataKey={`errors_${model.id}`}
                                stroke={COLORS[index % COLORS.length]}
                                name={`errors_${model.id}`}
                                strokeWidth={2}
                                dot={false}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <style>{`
                .model-comparison {
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }

                .chart-container {
                    margin-bottom: 30px;
                }

                h3 {
                    margin-top: 0;
                    margin-bottom: 20px;
                }

                h4 {
                    margin-top: 0;
                    margin-bottom: 10px;
                    color: #666;
                }

                .error {
                    color: #f44336;
                    text-align: center;
                    padding: 20px;
                }
            `}</style>
        </div>
    );
}; 