"""
Helper functions for API interactions.
"""

import time
import functools
from typing import Callable, Any
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("pharma_ai.api_helpers")

def rate_limit(calls_per_second: float):
    """
    Decorator to rate limit function calls.
    
    Args:
        calls_per_second: Maximum number of calls per second
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        
        return wrapper
    return decorator


def retry_on_error(max_attempts: int = 3, wait_seconds: int = 2):
    """
    Decorator to retry function on error.
    
    Args:
        max_attempts: Maximum number of retry attempts
        wait_seconds: Seconds to wait between retries
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=wait_seconds, min=1, max=10),
        reraise=True
    )


def sanitize_query(query: str) -> str:
    """
    Sanitize user query for API calls.
    
    Args:
        query: User input query
    
    Returns:
        Sanitized query string
    """
    # Remove special characters
    query = query.strip()
    
    # Remove excessive whitespace
    query = " ".join(query.split())
    
    # Limit length
    max_length = 500
    if len(query) > max_length:
        query = query[:max_length]
        logger.warning(f"Query truncated to {max_length} characters")
    
    return query


def format_error_message(error: Exception, context: str = "") -> str:
    """
    Format error message for user display.
    
    Args:
        error: Exception object
        context: Additional context
    
    Returns:
        Formatted error message
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    message = f"Error: {error_type}"
    if context:
        message += f" in {context}"
    if error_msg:
        message += f" - {error_msg}"
    
    return message


def parse_api_response(response: Any, expected_keys: list) -> dict:
    """
    Parse and validate API response.
    
    Args:
        response: API response object
        expected_keys: List of expected keys in response
    
    Returns:
        Parsed response dictionary
    """
    if not isinstance(response, dict):
        raise ValueError("Invalid response format - expected dictionary")
    
    missing_keys = [key for key in expected_keys if key not in response]
    
    if missing_keys:
        logger.warning(f"Missing keys in API response: {missing_keys}")
    
    return response