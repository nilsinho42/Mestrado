import { useEffect, useState } from 'react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    LineChart,
    Line
} from 'recharts';
import { getCloudCosts, getCloudPerformance } from '../../services/models';

interface CloudMetric {
    date: string;
    requestCount: number;
    avgLatency: number;
    cost: number;
}

interface CloudPerformanceMetric {
    platform: string;
    avgLatency: number;
    totalRequests: number;
    totalCost: number;
}

const CloudComparison = () => {
    const [metrics, setMetrics] = useState<Record<string, CloudMetric[]>>({});
    const [performance, setPerformance] = useState<CloudPerformanceMetric[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                const [metricsData, performanceData] = await Promise.all([
                    getCloudCosts(),
                    getCloudPerformance()
                ]);

                if (metricsData) {
                    setMetrics(metricsData as Record<string, CloudMetric[]>);
                }
                if (performanceData) {
                    setPerformance(performanceData as CloudPerformanceMetric[]);
                }
                
                setError(null);
            } catch (err) {
                setError('Failed to load cloud comparison data');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    if (loading) return <div>Loading cloud comparison data...</div>;
    if (error) return <div className="error">{error}</div>;
    if (!performance.length) return <div>No cloud comparison data available</div>;

    // Calculate cost per request for each platform
    const costPerRequestData = performance.map(p => ({
        platform: p.platform,
        costPerRequest: p.totalRequests > 0 ? p.totalCost / p.totalRequests : 0
    }));

    // Get all unique dates
    const allDates = Array.from(new Set(
        Object.values(metrics)
            .flat()
            .map(m => m.date)
            .sort((a, b) => new Date(a).getTime() - new Date(b).getTime())
    ));

    // Prepare data for the line chart
    const dailyCostData = allDates.map(date => {
        const dataPoint: { date: string; [key: string]: number | string } = { date };
        Object.entries(metrics).forEach(([platform, data]) => {
            const metric = data.find(m => m.date === date);
            dataPoint[platform] = metric?.cost || 0;
        });
        return dataPoint;
    });

    return (
        <div className="cloud-comparison">
            {/* Daily Cost Trends */}
            <div className="chart-container">
                <h4>Daily Cost Trends</h4>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={dailyCostData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                            dataKey="date" 
                            tickFormatter={(date) => new Date(date).toLocaleDateString()}
                        />
                        <YAxis />
                        <Tooltip 
                            formatter={(value: number) => `$${value.toFixed(2)}`}
                            labelFormatter={(date) => new Date(date as string).toLocaleDateString()}
                        />
                        <Legend />
                        {Object.keys(metrics).map((platform, index) => (
                            <Line
                                key={platform}
                                type="monotone"
                                dataKey={platform}
                                stroke={['#8884d8', '#82ca9d', '#ffc658'][index % 3]}
                                name={`${platform} Cost`}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </div>

            {/* Total Cost Comparison */}
            <div className="chart-container">
                <h4>Total Cost Comparison</h4>
                <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={performance}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="platform" />
                        <YAxis />
                        <Tooltip 
                            formatter={(value: number) => `$${value.toFixed(2)}`}
                        />
                        <Legend />
                        <Bar 
                            dataKey="totalCost" 
                            name="Total Cost" 
                            fill="#8884d8" 
                        />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Cost per Request Comparison */}
            <div className="chart-container">
                <h4>Cost per Request</h4>
                <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={costPerRequestData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="platform" />
                        <YAxis />
                        <Tooltip 
                            formatter={(value: number) => `$${value.toFixed(4)}`}
                        />
                        <Legend />
                        <Bar 
                            dataKey="costPerRequest" 
                            name="Cost per Request" 
                            fill="#82ca9d" 
                        />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Performance Comparison */}
            <div className="chart-container">
                <h4>Performance Comparison</h4>
                <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={performance}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="platform" />
                        <YAxis />
                        <Tooltip 
                            formatter={(value: number) => `${value.toFixed(2)}ms`}
                        />
                        <Legend />
                        <Bar 
                            dataKey="avgLatency" 
                            name="Average Latency" 
                            fill="#ffc658" 
                        />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Detailed Comparison Table */}
            <div className="table-container">
                <h4>Detailed Comparison</h4>
                <table>
                    <thead>
                        <tr>
                            <th>Platform</th>
                            <th>Total Requests</th>
                            <th>Avg. Latency</th>
                            <th>Total Cost</th>
                            <th>Cost/Request</th>
                        </tr>
                    </thead>
                    <tbody>
                        {performance.map(p => (
                            <tr key={p.platform}>
                                <td>{p.platform}</td>
                                <td>{p.totalRequests.toLocaleString()}</td>
                                <td>{p.avgLatency.toFixed(2)}ms</td>
                                <td>${p.totalCost.toFixed(2)}</td>
                                <td>${(p.totalRequests > 0 ? p.totalCost / p.totalRequests : 0).toFixed(4)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <style>{`
                .cloud-comparison {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }

                .chart-container {
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }

                h4 {
                    margin-top: 0;
                    margin-bottom: 20px;
                    color: #666;
                }

                .table-container {
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    overflow-x: auto;
                }

                table {
                    width: 100%;
                    border-collapse: collapse;
                }

                th, td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #eee;
                }

                th {
                    background-color: #f8f9fa;
                    font-weight: 600;
                }

                tr:last-child td {
                    border-bottom: none;
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

export default CloudComparison; 