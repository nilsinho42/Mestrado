from enum import Enum
import time
from typing import Callable, Any
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Service calls are blocked
    HALF_OPEN = "half_open"  # Testing if service is back

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: int = 60,
        half_open_timeout: int = 30
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.last_test_time = None
    
    def can_execute(self) -> bool:
        """Check if the circuit breaker allows execution."""
        if self.state == CircuitState.CLOSED:
            return True
            
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
                self.last_test_time = time.time()
                return True
            return False
            
        if self.state == CircuitState.HALF_OPEN:
            if time.time() - self.last_test_time >= self.half_open_timeout:
                return True
            return False
            
        return True
    
    def record_success(self):
        """Record a successful execution."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
    
    def record_failure(self):
        """Record a failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

def circuit_breaker(breaker: CircuitBreaker):
    """Decorator to apply circuit breaker to a function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            if not breaker.can_execute():
                raise Exception("Circuit breaker is open")
            
            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
                
        return wrapper
    return decorator 