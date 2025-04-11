import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import ErrorBoundary from "./components/shared/ErrorBoundary";
import ComparisonPanel from './components/comparison/ComparisonPanel';

export const App: React.FC = () => {
    return (
        <ErrorBoundary>
            <Router>
                <Routes>
                    <Route path="/comparison" element={<ComparisonPanel />} />
                    <Route path="/" element={<Navigate to="/comparison" />} />
                </Routes>
            </Router>
        </ErrorBoundary>
    );
}; 