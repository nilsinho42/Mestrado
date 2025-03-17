import { api } from './api';

export interface Location {
    id: string;
    name: string;
    latitude: number;
    longitude: number;
    metrics: {
        personCount: number;
        vehicleCount: number;
        avgFlow: number;
        lastUpdated: string;
    };
}

export interface LocationUpdateData {
    personCount: number;
    vehicleCount: number;
    avgFlow: number;
}

export const getLocations = async (): Promise<Location[]> => {
    const response = await api.get('/locations');
    return response.data;
};

export const getLocation = async (id: string): Promise<Location> => {
    const response = await api.get(`/locations/${id}`);
    return response.data;
};

export const addLocation = async (data: Omit<Location, 'id' | 'metrics'>): Promise<Location> => {
    const response = await api.post('/locations', data);
    return response.data;
};

export const updateLocationMetrics = async (id: string, data: LocationUpdateData): Promise<Location> => {
    const response = await api.patch(`/locations/${id}/metrics`, data);
    return response.data;
};

export const deleteLocation = async (id: string): Promise<void> => {
    await api.delete(`/locations/${id}`);
}; 