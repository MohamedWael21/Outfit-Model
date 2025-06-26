# Outfit Model API

A Flask-based API for generating compatible clothing outfits using deep learning and feature extraction.

## Features

- Add clothing items with image and category (features are extracted automatically)
- Delete clothing items by ID
- Generate compatible outfits

## Requirements

- Python 3.10+
- See `requirements.txt` for Python dependencies

## Local Setup

1. **Clone the repository**
2. **Create a virtual environment**
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the API**
   ```sh
   python main.py
   ```
   The API will be available at `http://localhost:8000`

## Docker Setup

1. **Build and run with Docker Compose**
   ```sh
   docker-compose up --build
   ```
   The API will be available at `http://localhost:8000`

## API Endpoints

### Add Item

- **POST** `/api/v1/items`
- **Form Data:**
  - `image`: Image file (required)
  - `category`: Category name (required)
  - `id`: (optional) Item ID
- **Response:** `{ "message": "Item added successfully", "id": <item_id> }`
- **Example (using curl):**
  ```sh
  curl -X POST -F "image=@path/to/image.jpg" -F "category=shirt" http://localhost:8000/api/v1/items
  ```

### Delete Item

- **DELETE** `/api/v1/items/<item_id>`
- **Response:** `{ "message": "Item <item_id> deleted successfully" }`
- **Example (using curl):**
  ```sh
  curl -X DELETE http://localhost:8000/api/v1/items/12345
  ```

### Generate Outfit

- **POST** `/api/v1/outfit/generate`
- **JSON Body:** `{ "seed_item_id": <id>, "max_items": <n> }`
- **Response:** Outfit items and generation time

### Health Check

- **GET** `/health`

## Notes

- The app uses SQLite for item storage (file: `items.db`).
- Model weights must be present as `outfit_compatibility_model.h5` in the project root.

---

Feel free to contribute or open issues for improvements!
