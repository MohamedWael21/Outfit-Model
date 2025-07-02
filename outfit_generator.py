import concurrent.futures
import numpy as np
from functools import lru_cache
from cache import PrecomputedCompatibilityCache

class FastOutfitGenerator:
    """Optimized outfit generator with caching and smart search"""
    
    def __init__(self, compatibility_model, product_database, cache=None):
        self.compatibility_model = compatibility_model
        self.product_db = product_database
        self.cache = cache or PrecomputedCompatibilityCache(use_redis=False)
        
        # Pre-defined category combinations for faster outfit generation
        self.outfit_templates = {
            'blazer': ['shirt', 'pants', 'shoes'],
            'blouse': ['pants', 'shoes'],
            'body': ['shoes'],
            'dress': ['shoes', 'outwear'],
            'hat': ['top', 'pants', 'shoes'],
            'hoodie': ['pants', 'shoes'],
            'longsleeve': ['pants',  'shoes'],
            'outwear': ['shirt', 'pants', 'shoes'],
            'pants': ['blouse',  'shoes'],
            'polo': ['pants',  'shoes'],
            'shirt': ['pants',  'shoes'],
            'shoes': ['shirt', 'pants', 'top'],
            'shorts': ['polo', 'shoes'],
            'skirt': ['shirt', 'top', 'shoes'],
            't-shirt': ['shorts', 'shoes'],
            'top': ['skirt', 'shoes'],
            'undershirt': ['shirt', 'pants', 'shoes']
        }
        
    
    def generate_outfit_fast(self, seed_item_id, max_items=4):
        """Fast outfit generation using precomputed similarities and templates"""
        seed_item = self.product_db.get_item(seed_item_id)
        if not seed_item:
            raise ValueError(f"Item {seed_item_id} not found")
        
        # Choose appropriate template
        template = self.outfit_templates.get(seed_item['category'].lower(), None)
        
        if template is None:
            return{
                'items': [seed_item],
                'item_count': 1,
            }
            
        
        # Find template that includes seed item category
        seed_category = seed_item['category'].lower()
        
        
        # Start with seed item
        outfit_items = [seed_item]
        required_categories = [cat for cat in template if cat != seed_category]

        
        # Find compatible items for each required category
        for category in required_categories:
            if len(outfit_items) >= max_items:
                break
            
            compatible_item = self._find_compatible_item_fast(
                outfit_items, category,
            )

            if compatible_item:
                outfit_items.append(compatible_item)
        
        
        return {
            'items': outfit_items,
            'item_count': len(outfit_items),
        }
    
    def _find_compatible_item_fast(self, current_items, target_category):
        """Find compatible item using similarity search and caching"""
        if not current_items:
            # Just return any item from the category
            print("No current items, returning any item from the category")
            candidates = self.product_db.get_items_by_category(target_category, limit=10)
            return candidates[0] if candidates else None
        
        candidates = self.product_db.get_items_by_category(target_category, limit=2000)
        
        if not candidates:
            return None
        
        # Remove items already in outfit
        current_ids = {item['id'] for item in current_items}
        candidates = [item for item in candidates if item['id'] not in current_ids]


        if not candidates:
            return None
        
        # Find best compatible item using cached scores
        best_item = None
        best_score = 0
        # Use parallel processing for compatibility calculation
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_candidate = {}
            
            for candidate in candidates[:100]:  
                future = executor.submit(
                    self._calculate_compatibility_with_outfit,
                    candidate, current_items
                )
                future_to_candidate[future] = candidate
            
            for future in concurrent.futures.as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                try:
                    avg_score = future.result(timeout=1.0)  # 1 second timeout
                    if avg_score > best_score:
                        best_score = avg_score
                        best_item = candidate
                except:
                    continue
        
        return best_item 
    
    def _calculate_compatibility_with_outfit(self, candidate, current_items):
        """Calculate average compatibility with current outfit items"""
        scores = []
        
        for current_item in current_items:
            # Check cache first
            cached_score = self.cache.get_compatibility(
                candidate['id'], current_item['id']
            )
            
            if cached_score is not None:
                scores.append(cached_score)
            else:
                # Compute and cache the score
                score = self.compatibility_model.predict_compatibility(
                    current_item['features'],
                    candidate['features']
                )
                scores.append(score)
                
                # Cache the result
                self.cache.set_compatibility(
                    candidate['id'], current_item['id'], score
                )
        
        return np.mean(scores) if scores else 0.0
    
