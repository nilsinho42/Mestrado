export interface Location {
  id: string;
  name: string;
  description?: string;
  latitude: number;
  longitude: number;
  metrics?: {
    personCount: number;
    vehicleCount: number;
    avgFlow: number;
    lastUpdated: string;
  };
  createdAt: string;
  updatedAt: string;
} 