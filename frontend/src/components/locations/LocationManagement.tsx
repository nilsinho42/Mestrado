import { useState, useEffect } from 'react';
import MapView from '../map/MapView';
import { Location } from '../../types/location';
import { getLocations, createLocation, deleteLocation } from '../../services/locations';

interface LocationFormData {
    name: string;
    latitude: number;
    longitude: number;
    description?: string;
}

const LocationManagement = () => {
    const [locations, setLocations] = useState<Location[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [formData, setFormData] = useState<LocationFormData>({
        name: '',
        latitude: 0,
        longitude: 0,
    });

    const fetchLocations = async () => {
        try {
            const data = await getLocations();
            setLocations(data);
            setError(null);
        } catch (err) {
            setError('Failed to fetch locations');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchLocations();
    }, []);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            setLoading(true);
            await createLocation({
                ...formData,
                createdAt: new Date().toISOString(),
                updatedAt: new Date().toISOString(),
            });
            await fetchLocations();
            setFormData({ name: '', latitude: 0, longitude: 0 });
            setError(null);
        } catch (err) {
            setError('Failed to create location');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleLocationSelect = (location: Location) => {
        setFormData({
            name: location.name,
            latitude: location.latitude,
            longitude: location.longitude,
            description: location.description,
        });
    };

    const handleDelete = async (id: string) => {
        try {
            setLoading(true);
            await deleteLocation(id);
            await fetchLocations();
            setError(null);
        } catch (err) {
            setError('Failed to delete location');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    if (loading) return <div>Loading locations...</div>;
    if (error) return <div className="error">{error}</div>;

    return (
        <div className="location-management">
            <h2>Location Management</h2>
            
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    placeholder="Location Name"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                />
                <input
                    type="number"
                    placeholder="Latitude"
                    value={formData.latitude}
                    onChange={(e) => setFormData({ ...formData, latitude: parseFloat(e.target.value) })}
                />
                <input
                    type="number"
                    placeholder="Longitude"
                    value={formData.longitude}
                    onChange={(e) => setFormData({ ...formData, longitude: parseFloat(e.target.value) })}
                />
                <button type="submit">Add Location</button>
            </form>

            <MapView locations={locations} onLocationSelect={handleLocationSelect} />

            <div className="locations-list">
                <h3>Locations</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Latitude</th>
                            <th>Longitude</th>
                            <th>Metrics</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {locations.map((location) => (
                            <tr key={location.id}>
                                <td>{location.name}</td>
                                <td>{location.latitude}</td>
                                <td>{location.longitude}</td>
                                <td>
                                    {location.metrics && (
                                        <>
                                            <p>People: {location.metrics.personCount}</p>
                                            <p>Vehicles: {location.metrics.vehicleCount}</p>
                                            <p>Avg Flow: {location.metrics.avgFlow}/min</p>
                                        </>
                                    )}
                                </td>
                                <td>
                                    <button onClick={() => handleDelete(location.id)}>Delete</button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default LocationManagement; 