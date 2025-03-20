import { useState, useCallback } from 'react';
import { login as loginApi, register as registerApi, logout as logoutApi, setToken } from '../services/auth';

interface User {
    id: number;
    email: string;
    role: string;
}

interface AuthState {
    user: User | null;
    isAuthenticated: boolean;
}

export const useAuth = () => {
    const [state, setState] = useState<AuthState>(() => {
        const token = localStorage.getItem('token');
        const userStr = localStorage.getItem('user');
        const user = userStr ? JSON.parse(userStr) : null;
        return {
            user,
            isAuthenticated: !!token,
        };
    });

    const login = useCallback(async (email: string, password: string) => {
        try {
            const response = await loginApi({ email, password });
            localStorage.setItem('token', response.token);
            localStorage.setItem('user', JSON.stringify(response.user));
            setToken(response.token);
            setState({
                user: response.user,
                isAuthenticated: true,
            });
            return response;
        } catch (error) {
            throw error;
        }
    }, []);

    const register = useCallback(async (email: string, password: string, confirmPassword: string) => {
        try {
            const response = await registerApi({ email, password, confirmPassword });
            localStorage.setItem('token', response.token);
            localStorage.setItem('user', JSON.stringify(response.user));
            setToken(response.token);
            setState({
                user: response.user,
                isAuthenticated: true,
            });
            return response;
        } catch (error) {
            throw error;
        }
    }, []);

    const logout = useCallback(() => {
        logoutApi();
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        setState({
            user: null,
            isAuthenticated: false,
        });
    }, []);

    return {
        user: state.user,
        isAuthenticated: state.isAuthenticated,
        login,
        register,
        logout,
    };
}; 