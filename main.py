# type: ignore
from flask import Flask, request, jsonify
import time
import os
import tempfile
from werkzeug.utils import secure_filename
from outfit_generator import FastOutfitGenerator
from outfit_compatibility_model import OutfitCompatibilityModel
from  item_database import ItemDatabase
from feature_extractor import ClothingFeatureExtractor

from cache import PrecomputedCompatibilityCache

product_db = None
outfit_generator = None
compatibility_model = None

app = Flask(__name__)



def initialize_api():
    """Initialize API components"""
    global product_db, outfit_generator, compatibility_model
    
    print("Initializing API...")
    
    # Load your trained model
    compatibility_model = OutfitCompatibilityModel()
    compatibility_model.load_weights('outfit_compatibility_model.h5')
    
    
    
    # Initialize optimized database
    product_db = ItemDatabase(db_path='items.db')
    
    redis_host = os.getenv('REDIS_HOST', None)
    redis_username = os.getenv('REDIS_USERNAME', None)
    redis_password = os.getenv('REDIS_PASSWORD', None)
    redis_port = os.getenv('REDIS_PORT', None)
    
    use_redis = redis_host is not None and redis_port is not None and redis_username is not None and redis_password is not None
    
    
    # Initialize cache (try Redis, fallback to memory)
    cache = PrecomputedCompatibilityCache(use_redis=use_redis, redis_host=redis_host, redis_port=redis_port, redis_username=redis_username, redis_password=redis_password)
    
    # Initialize outfit generator
    outfit_generator = FastOutfitGenerator(compatibility_model, product_db, cache)
   
    
    print("API initialized successfully!")

initialize_api()


@app.route('/api/v1/outfit/generate', methods=['POST'])
def generate_outfit():
    """API endpoint to generate outfit"""
    try:
        data = request.json
        seed_item_id = data.get('seed_item_id', None)
        max_items = data.get('max_items', 4)
        if seed_item_id is None:
            return jsonify({'error': 'seed_item_id is required'}), 400
        
        # Generate outfit
        outfit = outfit_generator.generate_outfit_fast(
            seed_item_id, max_items=max_items
        )

        
        
        # Format response
        response = {
            'outfit': {
                'items': [
                    {
                        'id': item['id'],
                    }
                    for item in outfit['items']
                ],
                'item_count': outfit['item_count'],
            },
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.route('/api/v1/items', methods=['POST'])
def add_item():
    """Endpoint to add a new item with image and category"""
    try:
        if 'image' not in request.files or 'category' not in request.form:
            return jsonify({'error': 'Image file and category are required.'}), 400
        image_file = request.files['image']
        category = request.form['category']
        item_id = request.form.get('id', None)
        # Save image to a temporary file
        filename = secure_filename(image_file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            image_path = tmp.name
            image_file.save(image_path)
        # Extract features
        extractor = ClothingFeatureExtractor()
        product_data = {'image_path': image_path, 'category': category}
        features = extractor.extract_all_features(product_data)
        # Prepare item dict
        item = {
            'id': int(item_id) if item_id is not None else int(time.time() * 1000),
            'category': category,
            'features': features
        }
        product_db.add_item_batch([item])
        # Clean up temp file
        os.remove(image_path)
        return jsonify({'message': 'Item added successfully', 'id': item['id']}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/items/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    """Endpoint to delete an item by id"""
    try:
        if item_id is None:
            return jsonify({'error': 'id is required'}), 400
        product_db.delete_item(int(item_id))
        return jsonify({'message': f'Item {item_id} deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"message": "API is running"})


if __name__ == '__main__':
    app.run()