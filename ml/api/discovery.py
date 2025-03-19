import consul
import socket
import logging
from typing import Optional, Dict, Any
import json
from .config import settings

logger = logging.getLogger(__name__)

class ServiceDiscovery:
    def __init__(self):
        self.consul = consul.Consul(
            host=settings.CONSUL_HOST,
            port=settings.CONSUL_PORT
        )
        self.service_id = f"{settings.SERVICE_NAME}-{socket.gethostname()}"
    
    async def register(self, port: int, tags: Optional[list] = None):
        """Register the service with Consul."""
        try:
            # Get the local IP address
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            
            # Prepare health check
            check = {
                "http": f"http://{ip_address}:{port}/health",
                "interval": "10s",
                "timeout": "5s"
            }
            
            # Register service
            self.consul.agent.service.register(
                name=settings.SERVICE_NAME,
                service_id=self.service_id,
                address=ip_address,
                port=port,
                tags=tags or [],
                check=check
            )
            
            logger.info(f"Registered service {self.service_id} with Consul")
            
        except Exception as e:
            logger.error(f"Failed to register service with Consul: {str(e)}")
            raise
    
    async def deregister(self):
        """Deregister the service from Consul."""
        try:
            self.consul.agent.service.deregister(self.service_id)
            logger.info(f"Deregistered service {self.service_id} from Consul")
        except Exception as e:
            logger.error(f"Failed to deregister service from Consul: {str(e)}")
            raise
    
    async def get_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get service details from Consul."""
        try:
            # Get service catalog
            _, services = self.consul.catalog.service(service_name)
            
            if not services:
                return None
            
            # Get the first healthy service
            service = services[0]
            
            return {
                "id": service["ServiceID"],
                "name": service["ServiceName"],
                "address": service["ServiceAddress"],
                "port": service["ServicePort"],
                "tags": service["ServiceTags"]
            }
            
        except Exception as e:
            logger.error(f"Failed to get service from Consul: {str(e)}")
            raise
    
    async def put_key(self, key: str, value: Any):
        """Store a key-value pair in Consul."""
        try:
            self.consul.kv.put(key, json.dumps(value))
        except Exception as e:
            logger.error(f"Failed to put key in Consul: {str(e)}")
            raise
    
    async def get_key(self, key: str) -> Optional[Any]:
        """Get a value from Consul by key."""
        try:
            _, data = self.consul.kv.get(key)
            if data is None:
                return None
            return json.loads(data["Value"])
        except Exception as e:
            logger.error(f"Failed to get key from Consul: {str(e)}")
            raise
    
    async def watch_key(self, key: str, callback):
        """Watch a key for changes."""
        index = None
        while True:
            try:
                index, data = self.consul.kv.get(
                    key,
                    index=index,
                    wait="30s"
                )
                if data is not None:
                    await callback(json.loads(data["Value"]))
            except Exception as e:
                logger.error(f"Error watching key: {str(e)}")
                # Continue watching even after error 