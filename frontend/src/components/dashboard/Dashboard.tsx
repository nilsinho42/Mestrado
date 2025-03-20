import React, { useState, useEffect } from 'react';
import { ModelComparison } from './ModelComparison';
import CloudComparison from './CloudComparison';
import { getModels } from '../../services/models';
import { Model } from '../../types/model';
import { UserMenu } from './UserMenu';

export const Dashboard: React.FC = () => {
    const [timeRange, setTimeRange] = useState<'24h' | '7d' | '30d'>('24h');
    const [models, setModels] = useState<Model[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchModels = async () => {
            try {
                setLoading(true);
                const data = await getModels({ status: 'active' });
                // Ensure data is an array even if API returns null
                setModels(data || []);
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

    return (
        <div className="dashboard">
            <header className="dashboard-header">
                <h2>Monitoring Dashboard</h2>
                <div className="header-right">
                    <UserMenu />
                </div>
            </header>

            <div className="dashboard-controls">
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
                ) : models.length === 0 ? (
                    <div className="no-models">No active models available</div>
                ) : (
                    <>
                        <div className="metrics-section">
                            <ModelComparison 
                                models={models}
                                timeRange={timeRange}
                            />
                        </div>

                        <div className="comparison-section">
                            <h3>Cloud Platform Comparison</h3>
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

                .header-right {
                    display: flex;
                    align-items: center;
                    gap: 2rem;
                }

                .dashboard-controls {
                    display: flex;
                    gap: 20px;
                    margin-bottom: 20px;
                }

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
                    display: flex;
                    flex-direction: column;
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
                .error,
                .no-models {
                    text-align: center;
                    padding: 20px;
                }

                .error {
                    color: #f44336;
                }

                .no-models {
                    color: #666;
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