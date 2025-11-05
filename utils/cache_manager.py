"""
Cache manager for API responses.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional
import logging
from config import CACHE_DIR, CACHE_TTL_HOURS, ENABLE_CACHING

logger = logging.getLogger("pharma_ai.cache")

class CacheManager:
    """Manages caching of API responses."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR, ttl_hours: int = CACHE_TTL_HOURS):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files
            ttl_hours: Time-to-live in hours
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.enabled = ENABLE_CACHING
    
    def _get_cache_key(self, source: str, query: str) -> str:
        """Generate cache key from source and query."""
        key_str = f"{source}:{query}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, source: str, query: str) -> Optional[Any]:
        """
        Retrieve cached data.
        
        Args:
            source: Data source name
            query: Query string
        
        Returns:
            Cached data or None if not found/expired
        """
        if not self.enabled:
            return None
        
        cache_key = self._get_cache_key(source, query)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            logger.debug(f"Cache miss for {source}:{query[:50]}")
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            # Check expiration
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cached_time > self.ttl:
                logger.debug(f"Cache expired for {source}:{query[:50]}")
                cache_path.unlink()
                return None
            
            logger.info(f"Cache hit for {source}:{query[:50]}")
            return cache_data['data']
        
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    def set(self, source: str, query: str, data: Any) -> None:
        """
        Store data in cache.
        
        Args:
            source: Data source name
            query: Query string
            data: Data to cache
        """
        if not self.enabled:
            return
        
        cache_key = self._get_cache_key(source, query)
        cache_path = self._get_cache_path(cache_key)
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'query': query,
            'data': data
        }
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.debug(f"Cached data for {source}:{query[:50]}")
        
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
    
    def clear(self, source: Optional[str] = None) -> int:
        """
        Clear cache files.
        
        Args:
            source: Optional source name to clear specific cache
        
        Returns:
            Number of files deleted
        """
        deleted = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                if source:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                    
                    if cache_data.get('source') == source:
                        cache_file.unlink()
                        deleted += 1
                else:
                    cache_file.unlink()
                    deleted += 1
            
            except Exception as e:
                logger.error(f"Error deleting cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {deleted} cache files")
        return deleted
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_files = len(list(self.cache_dir.glob("*.json")))
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))
        
        return {
            'total_files': total_files,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'enabled': self.enabled,
            'ttl_hours': CACHE_TTL_HOURS
        }

# Global cache instance
cache = CacheManager()