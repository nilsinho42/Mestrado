import React from 'react';
import { useAuth } from '../../hooks/useAuth';
import { useNavigate } from 'react-router-dom';

export const UserMenu: React.FC = () => {
    const { user, logout } = useAuth();
    const navigate = useNavigate();

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    return (
        <div className="user-menu">
            <span className="user-email">{user?.email}</span>
            <button onClick={handleLogout} className="logout-button">
                Logout
            </button>

            <style>{`
                .user-menu {
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                }

                .user-email {
                    color: #666;
                    font-size: 0.9rem;
                }

                .logout-button {
                    padding: 0.5rem 1rem;
                    background-color: #dc3545;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 0.9rem;
                    transition: background-color 0.2s;
                }

                .logout-button:hover {
                    background-color: #c82333;
                }
            `}</style>
        </div>
    );
}; 