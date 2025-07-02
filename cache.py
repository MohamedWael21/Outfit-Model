import redis
import threading
import pickle
class PrecomputedCompatibilityCache:
    
    """Cache for precomputed compatibility scores"""
    
    def __init__(self, redis_host=None, redis_port=None, redis_username=None, redis_password=None, use_redis=False):
        self.use_redis = use_redis
        if use_redis and redis_host and redis_port and redis_username and redis_password:
            try:
                self.redis_client = redis.Redis(host=redis_host, port=int(redis_port), username=redis_username, password=redis_password)
                self.redis_client.ping()
                print("Connected to Redis cache")
            except Exception as e:
                print(f"Redis connection failed: {e}")
                print("Redis not available, using in-memory cache")
                self.use_redis = False
        else:
            self.use_redis = False
            
        if not self.use_redis:
            self.memory_cache = {}
            self.cache_lock = threading.RLock()
    
    def _get_cache_key(self, item1_id, item2_id):
        """Generate cache key for item pair"""
        # Ensure consistent ordering
        if item1_id < item2_id:
            return f"compat:{item1_id}:{item2_id}"
        else:
            return f"compat:{item2_id}:{item1_id}"
    
    def get_compatibility(self, item1_id, item2_id):
        """Get cached compatibility score"""
        key = self._get_cache_key(item1_id, item2_id)
        
        if self.use_redis:
            cached = self.redis_client.get(key)
            if cached:
                return pickle.loads(cached)  # type: ignore
        else:
            with self.cache_lock:
                return self.memory_cache.get(key)
        
        return None
    
    def set_compatibility(self, item1_id, item2_id, score):
        """Cache compatibility score"""
        key = self._get_cache_key(item1_id, item2_id)
        
        if self.use_redis:
            self.redis_client.setex(key, 3600, pickle.dumps(score))  # 1 hour expiry
        else:
            with self.cache_lock:
                self.memory_cache[key] = score
    
    def batch_set_compatibility(self, compatibility_data):
        """Batch set compatibility scores"""
        if self.use_redis:
            pipe = self.redis_client.pipeline()
            for (item1_id, item2_id), score in compatibility_data.items():
                key = self._get_cache_key(item1_id, item2_id)
                pipe.setex(key, 3600, pickle.dumps(score))
            pipe.execute()
        else:
            with self.cache_lock:
                for (item1_id, item2_id), score in compatibility_data.items():
                    key = self._get_cache_key(item1_id, item2_id)
                    self.memory_cache[key] = score
