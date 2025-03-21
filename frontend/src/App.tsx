import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Login } from './components/auth/Login';
import { Dashboard } from './components/dashboard/Dashboard';
import ErrorBoundary from "./components/shared/ErrorBoundary";
import { useAuth } from './hooks/useAuth';
import RegisterForm from './components/RegisterForm';
import ComparisonPanel from './components/comparison/ComparisonPanel';

const PrivateRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const { isAuthenticated } = useAuth();
    return isAuthenticated ? <>{children}</> : <Navigate to="/login" />;
};

export const App: React.FC = () => {
    return (
        <ErrorBoundary>
            <Router>
                <Routes>
                    <Route path="/login" element={<Login />} />
                    <Route path="/register" element={<RegisterForm />} />
                    <Route
                        path="/dashboard"
                        element={
                            <PrivateRoute>
                                <Dashboard />
                            </PrivateRoute>
                        }
                    />
                    <Route
                        path="/comparison"
                        element={
                            <PrivateRoute>
                                <ComparisonPanel />
                            </PrivateRoute>
                        }
                    />
                    <Route path="/" element={<Navigate to="/dashboard" />} />
                </Routes>
            </Router>
        </ErrorBoundary>
    );
}; 