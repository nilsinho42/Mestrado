import api from './api';

interface LoginCredentials {
    email: string;
    password: string;
}

interface RegisterData extends LoginCredentials {
    confirmPassword: string;
}

interface AuthResponse {
    token: string;
    user: {
        id: number;
        email: string;
        role: string;
    };
}

export const login = async (credentials: LoginCredentials): Promise<AuthResponse> => {
    const response = await api.post<AuthResponse>('/auth/login', credentials);
    localStorage.setItem('token', response.data.token);
    return response.data;
};

export const register = async (data: RegisterData): Promise<AuthResponse> => {
    if (data.password !== data.confirmPassword) {
        throw new Error('Passwords do not match');
    }
    
    const { confirmPassword, ...registerData } = data;
    const response = await api.post<AuthResponse>('/auth/register', registerData);
    localStorage.setItem('token', response.data.token);
    return response.data;
};

export const logout = () => {
    localStorage.removeItem('token');
    window.location.href = '/login';
};

export const isAuthenticated = (): boolean => {
    return !!localStorage.getItem('token');
}; 