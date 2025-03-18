import api from './api';
import { Location } from '../types/location';

export const getLocations = async (): Promise<Location[]> => {
    const response = await api.get('/locations');
    return response.data;
};

export const getLocation = async (id: string): Promise<Location> => {
    const response = await api.get(`/locations/${id}`);
    return response.data;
};

export const createLocation = async (location: Omit<Location, 'id'>): Promise<Location> => {
    const response = await api.post('/locations', location);
    return response.data;
};

export const updateLocation = async (id: string, location: Partial<Location>): Promise<Location> => {
    const response = await api.put(`/locations/${id}`, location);
    return response.data;
};

export const deleteLocation = async (id: string): Promise<void> => {
    await api.delete(`/locations/${id}`);
}; 