import { useEffect, useState } from 'react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer
} from 'recharts';
import { getCloudCosts } from '../../services/models';

interface ComparisonData {
    platform: string;
    totalCost: number;
    totalRequests: number;
    avgLatency: number;
    costPerRequest: number;
}

const CloudComparison = () => {
    const [data, setData] = useState<ComparisonData[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                const costs = await getCloudCosts();

                // Process the data
                const processedData: ComparisonData[] = Object.entries(costs).map(([platform, metrics]) => ({
                    platform,
                    totalCost: metrics.totalCost,
                    totalRequests: metrics.totalRequests,
                    avgLatency: metrics.avgLatency,
                    costPerRequest: metrics.totalRequests > 0 
                        ? metrics.totalCost / metrics.totalRequests 
                        : 0
                }));

                setData(processedData);
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

    if (loading) return <div>Loading comparison data...</div>;
    if (error) return <div className="error">{error}</div>;
    if (!data.length) return <div>No comparison data available</div>;

    return (
        <div className="cloud-comparison">
            <h3>Cloud Platform Comparison</h3>

            {/* Cost Comparison */}
            <div className="chart-container">
                <h4>Total Cost Comparison</h4>
                <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="platform" />
                        <YAxis />
                        <Tooltip 
                            formatter={(value: number) => `$${value.toFixed(2)}`}
                        />
                        <Legend />
                        <Bar 
                            dataKey="totalCost" 
                            fill="#8884d8" 
                            name="Total Cost"
                        />
                        <Bar 
                            dataKey="costPerRequest" 
                            fill="#82ca9d" 
                            name="Cost per Request"
                        />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Performance Comparison */}
            <div className="chart-container">
                <h4>Performance Comparison</h4>
                <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="platform" />
                        <YAxis yAxisId="left" />
                        <YAxis yAxisId="right" orientation="right" />
                        <Tooltip />
                        <Legend />
                        <Bar 
                            yAxisId="left"
                            dataKey="avgLatency" 
                            fill="#8884d8" 
                            name="Avg. Latency (ms)"
                        />
                        <Bar 
                            yAxisId="right"
                            dataKey="totalRequests" 
                            fill="#82ca9d" 
                            name="Total Requests"
                        />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Summary Table */}
            <div className="comparison-table">
                <h4>Detailed Comparison</h4>
                <table>
                    <thead>
                        <tr>
                            <th>Platform</th>
                            <th>Total Cost</th>
                            <th>Total Requests</th>
                            <th>Avg. Latency</th>
                            <th>Cost per Request</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.map((item) => (
                            <tr key={item.platform}>
                                <td>{item.platform}</td>
                                <td>${item.totalCost.toFixed(2)}</td>
                                <td>{item.totalRequests.toLocaleString()}</td>
                                <td>{item.avgLatency.toFixed(2)}ms</td>
                                <td>${item.costPerRequest.toFixed(4)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default CloudComparison; 