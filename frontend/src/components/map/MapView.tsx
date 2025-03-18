import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { Location } from '../../types/location';

interface MapViewProps {
    locations: Location[];
    onLocationSelect?: (location: Location) => void;
}

const MapView: React.FC<MapViewProps> = ({ locations, onLocationSelect }) => {
    const handleMarkerClick = (locationId: string) => {
        onLocationSelect && onLocationSelect(locations.find(l => l.id === locationId) as Location);
    };

    return (
        <div className="map-container">
            <MapContainer
                center={[0, 0]}
                zoom={2}
                style={{ height: '600px', width: '100%' }}
            >
                <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                />
                {locations.map((location) => (
                    <Marker
                        key={location.id}
                        position={[location.latitude, location.longitude]}
                        eventHandlers={{
                            click: () => handleMarkerClick(location.id),
                        }}
                    >
                        <Popup>
                            <div className="popup-content">
                                <h3>{location.name}</h3>
                                {location.metrics && (
                                    <>
                                        <p>People: {location.metrics.personCount}</p>
                                        <p>Vehicles: {location.metrics.vehicleCount}</p>
                                        <p>Avg Flow: {location.metrics.avgFlow}/min</p>
                                    </>
                                )}
                            </div>
                        </Popup>
                    </Marker>
                ))}
            </MapContainer>

            <style>{`
                .map-container {
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    margin: 20px 0;
                }

                .popup-content {
                    padding: 10px;
                }

                .popup-content h3 {
                    margin: 0 0 10px 0;
                    color: #333;
                }

                .popup-content p {
                    margin: 5px 0;
                    color: #666;
                }
            `}</style>
        </div>
    );
};

export default MapView; 