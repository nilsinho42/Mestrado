import React, { useEffect, useState } from 'react';

interface ToastProps {
    message: string;
    type?: 'success' | 'error' | 'info' | 'warning';
    duration?: number;
    onClose?: () => void;
}

export const Toast: React.FC<ToastProps> = ({
    message,
    type = 'info',
    duration = 3000,
    onClose,
}) => {
    const [isVisible, setIsVisible] = useState(true);

    useEffect(() => {
        const timer = setTimeout(() => {
            setIsVisible(false);
            onClose?.();
        }, duration);

        return () => clearTimeout(timer);
    }, [duration, onClose]);

    if (!isVisible) return null;

    return (
        <div className={`toast toast-${type}`}>
            <div className="toast-content">
                {message}
                <button className="close-button" onClick={() => setIsVisible(false)}>
                    ×
                </button>
            </div>

            <style>{`
                .toast {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    z-index: 1000;
                    min-width: 300px;
                    max-width: 500px;
                    padding: 1rem;
                    border-radius: 4px;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
                    animation: slideIn 0.3s ease-out;
                }

                .toast-content {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }

                .close-button {
                    background: none;
                    border: none;
                    color: inherit;
                    font-size: 1.5rem;
                    cursor: pointer;
                    padding: 0 0.5rem;
                    margin-left: 1rem;
                    opacity: 0.7;
                }

                .close-button:hover {
                    opacity: 1;
                }

                .toast-success {
                    background-color: #d4edda;
                    color: #155724;
                    border: 1px solid #c3e6cb;
                }

                .toast-error {
                    background-color: #f8d7da;
                    color: #721c24;
                    border: 1px solid #f5c6cb;
                }

                .toast-info {
                    background-color: #cce5ff;
                    color: #004085;
                    border: 1px solid #b8daff;
                }

                .toast-warning {
                    background-color: #fff3cd;
                    color: #856404;
                    border: 1px solid #ffeeba;
                }

                @keyframes slideIn {
                    from {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }
            `}</style>
        </div>
    );
}; 