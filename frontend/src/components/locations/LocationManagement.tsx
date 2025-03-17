import React, { useEffect, useState } from 'react';
import { MapView } from '../map/MapView';
import { Location, getLocations, addLocation, deleteLocation } from '../../services/locations';

interface LocationFormData {
    name: string;
    latitude: number;
    longitude: number;
}

export const LocationManagement: React.FC = () => {
    const [locations, setLocations] = useState<Location[]>([]);
    const [formData, setFormData] = useState<LocationFormData>({
        name: '',
        latitude: 0,
        longitude: 0,
    });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetchLocations();
    }, []);

    const fetchLocations = async () => {
        try {
            setLoading(true);
            const data = await getLocations();
            setLocations(data);
            setError(null);
        } catch (err) {
            setError('Failed to fetch locations');
        } finally {
            setLoading(false);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            setLoading(true);
            await addLocation(formData);
            await fetchLocations();
            setFormData({ name: '', latitude: 0, longitude: 0 });
            setError(null);
        } catch (err) {
            setError('Failed to add location');
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async (id: string) => {
        try {
            setLoading(true);
            await deleteLocation(id);
            await fetchLocations();
            setError(null);
        } catch (err) {
            setError('Failed to delete location');
        } finally {
            setLoading(false);
        }
    };

    const handleLocationSelect = (locationId: string) => {
        const location = locations.find(loc => loc.id === locationId);
        if (location) {
            setFormData({
                name: location.name,
                latitude: location.latitude,
                longitude: location.longitude,
            });
        }
    };

    return (
        <div className="location-management">
            <h2>Location Management</h2>
            
            <div className="location-form">
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label htmlFor="name">Location Name:</label>
                        <input
                            type="text"
                            id="name"
                            value={formData.name}
                            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label htmlFor="latitude">Latitude:</label>
                        <input
                            type="number"
                            id="latitude"
                            step="any"
                            value={formData.latitude}
                            onChange={(e) => setFormData({ ...formData, latitude: parseFloat(e.target.value) })}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label htmlFor="longitude">Longitude:</label>
                        <input
                            type="number"
                            id="longitude"
                            step="any"
                            value={formData.longitude}
                            onChange={(e) => setFormData({ ...formData, longitude: parseFloat(e.target.value) })}
                            required
                        />
                    </div>
                    <button type="submit" disabled={loading}>
                        Add Location
                    </button>
                </form>
            </div>

            {error && <div className="error-message">{error}</div>}

            <div className="map-section">
                <MapView
                    locations={locations}
                    onLocationSelect={handleLocationSelect}
                />
            </div>

            <div className="locations-list">
                <h3>Monitoring Locations</h3>
                {locations.map((location) => (
                    <div key={location.id} className="location-item">
                        <div className="location-info">
                            <h4>{location.name}</h4>
                            <p>Lat: {location.latitude}, Long: {location.longitude}</p>
                            <p>People: {location.metrics.personCount}</p>
                            <p>Vehicles: {location.metrics.vehicleCount}</p>
                            <p>Flow: {location.metrics.avgFlow}/min</p>
                        </div>
                        <button
                            onClick={() => handleDelete(location.id)}
                            className="delete-button"
                            disabled={loading}
                        >
                            Delete
                        </button>
                    </div>
                ))}
            </div>

            <style>{`
                .location-management {
                    padding: 20px;
                }

                .location-form {
                    max-width: 500px;
                    margin-bottom: 20px;
                }

                .form-group {
                    margin-bottom: 15px;
                }

                .form-group label {
                    display: block;
                    margin-bottom: 5px;
                }

                .form-group input {
                    width: 100%;
                    padding: 8px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }

                .error-message {
                    color: #ff4444;
                    margin: 10px 0;
                }

                .map-section {
                    margin: 20px 0;
                }

                .locations-list {
                    margin-top: 20px;
                }

                .location-item {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    margin-bottom: 10px;
                }

                .location-info h4 {
                    margin: 0 0 10px 0;
                }

                .location-info p {
                    margin: 5px 0;
                    color: #666;
                }

                .delete-button {
                    background-color: #ff4444;
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    border-radius: 4px;
                    cursor: pointer;
                }

                .delete-button:disabled {
                    background-color: #ffaaaa;
                    cursor: not-allowed;
                }

                button {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 4px;
                    cursor: pointer;
                }

                button:disabled {
                    background-color: #8bc98e;
                    cursor: not-allowed;
                }
            `}</style>
        </div>
    );
}; 