import { useEffect, useRef, useState, useCallback } from 'react';

interface WebSocketMessage {
    type: string;
    data: any;
}

export const useWebSocket = (url: string) => {
    const ws = useRef<WebSocket | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

    const connect = useCallback(() => {
        ws.current = new WebSocket(url);

        ws.current.onopen = () => {
            setIsConnected(true);
            console.log('WebSocket connected');
        };

        ws.current.onclose = () => {
            setIsConnected(false);
            console.log('WebSocket disconnected');
            // Attempt to reconnect after 3 seconds
            setTimeout(connect, 3000);
        };

        ws.current.onerror = (error) => {
            console.error('WebSocket error:', error);
            ws.current?.close();
        };

        ws.current.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                setLastMessage(message);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };
    }, [url]);

    const sendMessage = useCallback((type: string, data: any) => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify({ type, data }));
        } else {
            console.error('WebSocket is not connected');
        }
    }, []);

    useEffect(() => {
        connect();

        return () => {
            if (ws.current) {
                ws.current.close();
            }
        };
    }, [connect]);

    return {
        isConnected,
        lastMessage,
        sendMessage,
    };
}; 