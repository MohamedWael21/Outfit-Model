import sqlite3
import numpy as np
import pickle

class ItemDatabase:
    """Optimized item database with precomputed features and indexing"""
    
    def __init__(self, db_path="items.db"):
        self.db_path = db_path
        self.feature_dim = 200
        
        # Initialize database
        self._init_database()
        
        # Load items into memory for fast access
        self.items_cache = {}
        self.category_cache = {}
        self._load_cache()
    
    def _init_database(self):
        """Initialize SQLite database for items"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY,
                category TEXT,
                features BLOB
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_category ON items(category)
        ''')
        
        conn.commit()
        conn.close()
    
    def add_item_batch(self, items):
        """Add multiple items efficiently"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data_to_insert = []
        for item in items:
            # Normalize features for cosine similarity
            features = item['features']
            normalized_features = features / (np.linalg.norm(features) + 1e-8)
            
            data_to_insert.append((
                item['id'],
                item['category'],
                pickle.dumps(normalized_features)
            ))
        
        cursor.executemany('''
            INSERT OR REPLACE INTO items 
            (id, category, features)
            VALUES (?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
        conn.close()
        
        # Rebuild cache and FAISS index
        self._load_cache()
    
    def _load_cache(self):
        """Load items into memory cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, category, features FROM items')
        rows = cursor.fetchall()
        
        self.items_cache.clear()
        self.category_cache.clear()
        
        for row in rows:
            item_id, category, features_blob = row
            features = pickle.loads(features_blob)
            
            item_data = {
                'id': item_id,
                'category': category,
                'features': features
            }
            
            self.items_cache[item_id] = item_data
            
            if category not in self.category_cache:
                self.category_cache[category] = []
            self.category_cache[category].append(item_data)
        
        conn.close()
        print(f"Loaded {len(self.items_cache)} items into cache")

    def delete_item(self, item_id):
        """Delete item from cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM items WHERE id = ?', (item_id,))
        conn.commit()
        conn.close()
        self._load_cache()
    
    def get_item(self, item_id):
        """Get item from cache"""
        return self.items_cache.get(item_id)
    
    def get_items_by_category(self, category, limit=50):
        """Get items by category with limit"""
        items = self.category_cache.get(category.lower(), [])
        return items[:limit]
