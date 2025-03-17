import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
    children: ReactNode;
    fallback?: ReactNode;
}

interface State {
    hasError: boolean;
    error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
    public state: State = {
        hasError: false,
        error: null,
    };

    public static getDerivedStateFromError(error: Error): State {
        return {
            hasError: true,
            error,
        };
    }

    public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error('Uncaught error:', error, errorInfo);
    }

    public render() {
        if (this.state.hasError) {
            return this.props.fallback || (
                <div className="error-boundary">
                    <h2>Something went wrong</h2>
                    <details>
                        <summary>Error details</summary>
                        <pre>{this.state.error?.toString()}</pre>
                    </details>
                    <button onClick={() => window.location.reload()}>
                        Refresh Page
                    </button>

                    <style>{`
                        .error-boundary {
                            padding: 2rem;
                            text-align: center;
                            background-color: #fff3f3;
                            border-radius: 8px;
                            margin: 2rem;
                        }

                        .error-boundary h2 {
                            color: #dc3545;
                            margin-bottom: 1rem;
                        }

                        .error-boundary details {
                            margin: 1rem 0;
                            padding: 1rem;
                            background-color: #f8f9fa;
                            border-radius: 4px;
                            text-align: left;
                        }

                        .error-boundary pre {
                            margin-top: 1rem;
                            white-space: pre-wrap;
                            word-break: break-all;
                        }

                        .error-boundary button {
                            padding: 0.5rem 1rem;
                            background-color: #007bff;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            cursor: pointer;
                            transition: background-color 0.2s;
                        }

                        .error-boundary button:hover {
                            background-color: #0056b3;
                        }
                    `}</style>
                </div>
            );
        }

        return this.props.children;
    }
} 