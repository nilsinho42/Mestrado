/// <reference types="vite/client" />

import axios from 'axios';

const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8080',
    headers: {
        'Content-Type': 'application/json',
    },
    // Set default timeout to 2 minutes
    timeout: 120000,
});

// Add request interceptor for logging
api.interceptors.request.use(
    (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        
        // For file uploads, use a longer timeout
        const url = config.url || '';
        const contentType = String(config.headers?.['Content-Type'] || '');
        
        if (url.includes('upload') && contentType.includes('multipart/form-data')) {
            config.timeout = 300000; // 5 minutes for uploads
        }
        
        return config;
    },
    (error) => {
        console.error('Request error:', error);
        return Promise.reject(error);
    }
);

// Enhanced error handling interceptor
api.interceptors.response.use(
    (response) => {
        console.log(`API Response: ${response.status} ${response.config.url}`);
        return response;
    },
    async (error) => {
        if (error.response) {
            // The request was made and the server responded with an error status
            console.error('API Error Response:', {
                status: error.response.status,
                data: error.response.data,
                url: error.config?.url
            });
        } else if (error.request) {
            // The request was made but no response was received
            console.error('API No Response:', {
                request: error.request,
                timeout: error.config?.timeout,
                url: error.config?.url
            });
        } else {
            // Something happened in setting up the request
            console.error('API Request Setup Error:', error.message);
        }
        
        return Promise.reject(error);
    }
);

export default api; 