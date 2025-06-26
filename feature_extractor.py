import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from skimage.feature import local_binary_pattern

class ClothingFeatureExtractor:
    """Extract features from clothing items for compatibility analysis"""

    def __init__(self):
        self.color_bins = 32
        self.category_encoder = LabelEncoder()
        # Updated category list
        self.categories = ['blazer', 'blouse', 'body', 'dress', 'hat', 'hoodie',
                          'longsleeve', 'outwear', 'pants', 'polo', 'shirt', 'shoes',
                          'shorts', 'skirt', 't-shirt', 'top', 'undershirt']

    def extract_color_features(self, image_path):
        """Extract color histogram features"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract color histograms for each channel
            hist_r = cv2.calcHist([image], [0], None, [self.color_bins], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [self.color_bins], [0, 256])
            hist_b = cv2.calcHist([image], [2], None, [self.color_bins], [0, 256])

            # Normalize histograms
            hist_r = hist_r.flatten() / (np.sum(hist_r) + 1e-8)
            hist_g = hist_g.flatten() / (np.sum(hist_g) + 1e-8)
            hist_b = hist_b.flatten() / (np.sum(hist_b) + 1e-8)

            # Extract dominant colors (top 3 colors per channel)
            dominant_features = []
            for hist in [hist_r, hist_g, hist_b]:
                top_indices = np.argsort(hist)[-3:]
                dominant_features.extend(hist[top_indices])

            return np.concatenate([hist_r, hist_g, hist_b, dominant_features])

        except Exception as e:
            print(f"Error extracting color features from {image_path}: {e}")
            # Return zero features if image cannot be processed
            return np.zeros(self.color_bins * 3 + 9)

    def extract_texture_features(self, image_path):
        """Extract texture features using Local Binary Patterns and edge detection"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Resize image to standard size for consistent feature extraction
            image = cv2.resize(image, (224, 224))

            # Local Binary Pattern
            def get_lbp(image, radius=3, n_points=24):
                try:
                    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
                    hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2,
                                         range=(0, n_points + 2), density=True)
                    return hist
                except ImportError:
                    # Fallback if scikit-image is not available
                    return np.zeros(n_points + 2)

            lbp_features = get_lbp(image)

            # Edge detection features
            edges = cv2.Canny(image, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

            # Texture contrast and uniformity
            gray_hist = cv2.calcHist([image], [0], None, [32], [0, 256])
            gray_hist = gray_hist.flatten() / (np.sum(gray_hist) + 1e-8)

            # Statistical texture features
            mean_intensity = np.mean(image) # type: ignore
            std_intensity = np.std(image) # type: ignore

            texture_stats = [edge_density, mean_intensity / 255.0, std_intensity / 255.0]

            return np.concatenate([lbp_features, gray_hist, texture_stats])

        except Exception as e:
            print(f"Error extracting texture features from {image_path}: {e}")
            # Return zero features if image cannot be processed
            return np.zeros(26 + 32 + 3)  # LBP + hist + stats

    def extract_category_features(self, category):
        """Extract features from category information"""
        # Get category index
        category_lower = category.lower()
        try:
            category_idx = self.categories.index(category_lower)
        except ValueError:
            # If category not found, default to 'top' (index 15)
            category_idx = 15

        # Create one-hot encoding
        category_onehot = np.zeros(len(self.categories))  # 17 categories
        category_onehot[category_idx] = 1

        # Add category compatibility features
        compatibility_features = self._get_category_compatibility_features(category_idx)

        return np.concatenate([category_onehot, compatibility_features])

    def _get_category_compatibility_features(self, category_idx):
        """Get compatibility features based on category"""
        # Define which categories typically go together
        # Updated compatibility matrix for 17 categories
        compatibility_matrix = np.array([
            # blazer, blouse, body, dress, hat, hoodie, longsleeve, outwear, pants, polo, shirt, shoes, shorts, skirt, t-shirt, top, undershirt
            [0.5, 0.7, 0.3, 0.2, 0.6, 0.4, 0.6, 0.8, 0.9, 0.8, 0.8, 0.8, 0.7, 0.9, 0.7, 0.8, 0.3],  # blazer
            [0.7, 0.5, 0.3, 0.2, 0.6, 0.3, 0.6, 0.6, 0.9, 0.7, 0.8, 0.8, 0.7, 0.9, 0.6, 0.8, 0.3],  # blouse
            [0.3, 0.3, 0.5, 0.2, 0.4, 0.6, 0.7, 0.5, 0.8, 0.6, 0.7, 0.7, 0.8, 0.8, 0.8, 0.7, 0.6],  # body
            [0.2, 0.2, 0.2, 0.5, 0.8, 0.3, 0.3, 0.7, 0.1, 0.2, 0.2, 0.9, 0.1, 0.1, 0.2, 0.2, 0.1],  # dress
            [0.6, 0.6, 0.4, 0.8, 0.5, 0.6, 0.6, 0.7, 0.6, 0.6, 0.6, 0.3, 0.6, 0.6, 0.6, 0.6, 0.4],  # hat
            [0.4, 0.3, 0.6, 0.3, 0.6, 0.5, 0.7, 0.6, 0.8, 0.7, 0.6, 0.8, 0.8, 0.6, 0.8, 0.7, 0.4],  # hoodie
            [0.6, 0.6, 0.7, 0.3, 0.6, 0.7, 0.5, 0.6, 0.8, 0.7, 0.8, 0.8, 0.7, 0.8, 0.8, 0.8, 0.5],  # longsleeve
            [0.8, 0.6, 0.5, 0.7, 0.7, 0.6, 0.6, 0.5, 0.7, 0.6, 0.7, 0.6, 0.6, 0.7, 0.6, 0.7, 0.4],  # outwear
            [0.9, 0.9, 0.8, 0.1, 0.6, 0.8, 0.8, 0.7, 0.5, 0.8, 0.8, 0.8, 0.3, 0.3, 0.8, 0.8, 0.6],  # pants
            [0.8, 0.7, 0.6, 0.2, 0.6, 0.7, 0.7, 0.6, 0.8, 0.5, 0.8, 0.8, 0.7, 0.8, 0.8, 0.8, 0.5],  # polo
            [0.8, 0.8, 0.7, 0.2, 0.6, 0.6, 0.8, 0.7, 0.8, 0.8, 0.5, 0.8, 0.7, 0.8, 0.8, 0.8, 0.6],  # shirt
            [0.8, 0.8, 0.7, 0.9, 0.3, 0.8, 0.8, 0.6, 0.8, 0.8, 0.8, 0.5, 0.8, 0.8, 0.8, 0.8, 0.6],  # shoes
            [0.7, 0.7, 0.8, 0.1, 0.6, 0.8, 0.7, 0.6, 0.3, 0.7, 0.7, 0.8, 0.5, 0.3, 0.8, 0.8, 0.5],  # shorts
            [0.9, 0.9, 0.8, 0.1, 0.6, 0.6, 0.8, 0.7, 0.3, 0.8, 0.8, 0.8, 0.3, 0.5, 0.8, 0.8, 0.6],  # skirt
            [0.7, 0.6, 0.8, 0.2, 0.6, 0.8, 0.8, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.5, 0.8, 0.7],  # t-shirt
            [0.8, 0.8, 0.7, 0.2, 0.6, 0.7, 0.8, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.5, 0.6],  # top
            [0.3, 0.3, 0.6, 0.1, 0.4, 0.4, 0.5, 0.4, 0.6, 0.5, 0.6, 0.6, 0.5, 0.6, 0.7, 0.6, 0.5],  # undershirt
        ])

        return compatibility_matrix[category_idx]

    def extract_all_features(self, product_data):
        """Extract all features for a clothing item"""
        # Extract visual features
        color_features = self.extract_color_features(product_data['image_path'])
        texture_features = self.extract_texture_features(product_data['image_path'])

        # Extract category features
        category_features = self.extract_category_features(product_data['category'])

        # Combine all features
        all_features = np.concatenate([
            color_features,      # 105 features (32*3 + 9 dominant colors)
            texture_features,    # 61 features (26 LBP + 32 hist + 3 stats)
            category_features    # 34 features (17 onehot + 17 compatibility)
        ])

        return all_features  # Total: 200 features
