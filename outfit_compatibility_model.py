# type: ignore
import tensorflow as tf  
import numpy as np
class OutfitCompatibilityModel:
    """Neural network model for clothing compatibility prediction"""

    def __init__(self, feature_dim=200):  # Updated feature dimension
        self.feature_dim = feature_dim
        self.model = self._build_model()
        self.normalizer1 = tf.keras.layers.Normalization(axis=-1)
        self.normalizer2 = tf.keras.layers.Normalization(axis=-1)

    def _build_model(self):
        """Build the compatibility neural network"""

        # Input layers for two clothing items
        item1_input = tf.keras.layers.Input(shape=(self.feature_dim,), name='item1')
        item2_input = tf.keras.layers.Input(shape=(self.feature_dim,), name='item2')

        # Shared feature processing network
        shared_network = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
        ], name='shared_feature_network')

        # Process both items through shared network
        item1_processed = shared_network(item1_input)
        item2_processed = shared_network(item2_input)

        # Compatibility analysis layers
        # Method 1: Concatenation
        concatenated = tf.keras.layers.Concatenate()([item1_processed, item2_processed])

        # Method 2: Element-wise operations
        difference = tf.keras.layers.Subtract()([item1_processed, item2_processed])
        abs_difference = tf.keras.layers.Lambda(lambda x: tf.abs(x))(difference)

        product = tf.keras.layers.Multiply()([item1_processed, item2_processed])

        # Method 3: Cosine similarity
        item1_norm = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(item1_processed)
        item2_norm = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(item2_processed)
        cosine_sim = tf.keras.layers.Dot(axes=1)([item1_norm, item2_norm])
        cosine_sim = tf.keras.layers.Reshape((1,))(cosine_sim)

        # Combine all compatibility measures
        compatibility_features = tf.keras.layers.Concatenate()([
            concatenated,
            abs_difference,
            product,
            cosine_sim
        ])

        # Final compatibility prediction layers
        compatibility = tf.keras.layers.Dense(128, activation='relu')(compatibility_features)
        compatibility = tf.keras.layers.BatchNormalization()(compatibility)
        compatibility = tf.keras.layers.Dropout(0.3)(compatibility)

        compatibility = tf.keras.layers.Dense(64, activation='relu')(compatibility)
        compatibility = tf.keras.layers.BatchNormalization()(compatibility)
        compatibility = tf.keras.layers.Dropout(0.2)(compatibility)

        # Output layer - compatibility score between 0 and 1
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='compatibility_score')(compatibility)

        # Create model
        model = tf.keras.Model(
            inputs=[item1_input, item2_input],
            outputs=output
        )

        return model

    def load_weights(self, model_path):
        """Load the model from a file"""
        self.model.load_weights(model_path)

    def predict_compatibility(self, item1_features, item2_features):
        """Predict compatibility between two items"""
        item1_scaled = self.normalizer1(tf.convert_to_tensor([item1_features]))
        item2_scaled = self.normalizer2(tf.convert_to_tensor([item2_features]))

        compatibility_score = self.model.predict([item1_scaled, item2_scaled])[0][0]
        return compatibility_score

