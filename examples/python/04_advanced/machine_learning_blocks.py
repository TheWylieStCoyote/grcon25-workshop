#!/usr/bin/env python3
"""
Machine Learning Integration in GNU Radio
Advanced example showing ML-based signal processing
"""

import numpy as np
from gnuradio import gr
import pmt
import pickle
import os

# Optional: Import ML libraries if available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available, using numpy-based implementations")

try:
    from sklearn.cluster import KMeans
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available, using basic implementations")


class NeuralNetworkDemodulator(gr.sync_block):
    """
    Neural Network based demodulator
    Can learn to demodulate complex modulation schemes
    """
    
    def __init__(self, model_path=None, modulation_type='qpsk', 
                 training_mode=False):
        """
        Initialize NN demodulator
        
        Args:
            model_path: Path to pre-trained model
            modulation_type: Type of modulation to demodulate
            training_mode: Whether to collect training data
        """
        gr.sync_block.__init__(
            self,
            name='Neural Network Demodulator',
            in_sig=[np.complex64],
            out_sig=[np.uint8]  # Demodulated bits
        )
        
        self.modulation_type = modulation_type
        self.training_mode = training_mode
        
        # Model parameters
        self.input_size = 32  # IQ samples per decision
        self.hidden_size = 64
        self.output_size = 8  # Bits per symbol (max)
        
        # Initialize or load model
        if TF_AVAILABLE:
            if model_path and os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"Loaded model from {model_path}")
            else:
                self.model = self.build_model()
                print("Created new neural network model")
        else:
            # Fallback to numpy implementation
            self.model = None
            self.weights = self.initialize_weights()
        
        # Training data collection
        self.training_data = []
        self.training_labels = []
        
        # Performance metrics
        self.decisions_made = 0
        self.confidence_scores = []
    
    def build_model(self):
        """Build neural network model using TensorFlow"""
        if not TF_AVAILABLE:
            return None
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.hidden_size, 
                activation='relu',
                input_shape=(self.input_size * 2,)  # Real and imag parts
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.hidden_size, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.output_size, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def initialize_weights(self):
        """Initialize weights for numpy-based NN"""
        np.random.seed(42)
        return {
            'W1': np.random.randn(self.input_size * 2, self.hidden_size) * 0.1,
            'b1': np.zeros(self.hidden_size),
            'W2': np.random.randn(self.hidden_size, self.hidden_size) * 0.1,
            'b2': np.zeros(self.hidden_size),
            'W3': np.random.randn(self.hidden_size, self.output_size) * 0.1,
            'b3': np.zeros(self.output_size)
        }
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward_pass(self, x):
        """Forward pass through numpy NN"""
        # Layer 1
        z1 = np.dot(x, self.weights['W1']) + self.weights['b1']
        a1 = self.relu(z1)
        
        # Layer 2
        z2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
        a2 = self.relu(z2)
        
        # Output layer
        z3 = np.dot(a2, self.weights['W3']) + self.weights['b3']
        output = self.sigmoid(z3)
        
        return output
    
    def work(self, input_items, output_items):
        """
        Demodulate using neural network
        """
        input_samples = input_items[0]
        output_bits = output_items[0]
        
        # Process in chunks
        num_chunks = len(input_samples) // self.input_size
        
        for i in range(num_chunks):
            # Extract chunk
            start_idx = i * self.input_size
            end_idx = start_idx + self.input_size
            chunk = input_samples[start_idx:end_idx]
            
            # Prepare input (separate real and imaginary)
            nn_input = np.concatenate([chunk.real, chunk.imag])
            
            # Get prediction
            if TF_AVAILABLE and self.model:
                # TensorFlow prediction
                prediction = self.model.predict(
                    nn_input.reshape(1, -1), 
                    verbose=0
                )[0]
            else:
                # Numpy prediction
                prediction = self.forward_pass(nn_input)
            
            # Convert to bits
            bits = (prediction > 0.5).astype(np.uint8)
            
            # Pack bits into bytes
            if i < len(output_bits):
                output_bits[i] = np.packbits(bits)[0]
            
            # Track confidence
            confidence = np.mean(np.abs(prediction - 0.5) * 2)
            self.confidence_scores.append(confidence)
            
            # Collect training data if in training mode
            if self.training_mode:
                self.training_data.append(nn_input)
                # In real scenario, labels would come from known data
                self.training_labels.append(bits)
            
            self.decisions_made += 1
        
        return num_chunks
    
    def train_model(self, epochs=10):
        """Train the model on collected data"""
        if not self.training_data:
            print("No training data collected")
            return
        
        X = np.array(self.training_data)
        y = np.array(self.training_labels)
        
        if TF_AVAILABLE and self.model:
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
            return history
        else:
            print("Training not implemented for numpy version")
            return None
    
    def save_model(self, path):
        """Save trained model"""
        if TF_AVAILABLE and self.model:
            self.model.save(path)
            print(f"Model saved to {path}")
        else:
            with open(path, 'wb') as f:
                pickle.dump(self.weights, f)
            print(f"Weights saved to {path}")


class AutomaticModulationClassifier(gr.sync_block):
    """
    Automatic Modulation Classification using ML
    Identifies modulation type from received signal
    """
    
    def __init__(self, feature_length=1024, classifier_type='svm'):
        """
        Initialize AMC
        
        Args:
            feature_length: Length of signal segment for classification
            classifier_type: Type of classifier ('svm', 'kmeans', 'nn')
        """
        gr.sync_block.__init__(
            self,
            name='Automatic Modulation Classifier',
            in_sig=[np.complex64],
            out_sig=[np.float32]  # Classification confidence
        )
        
        self.feature_length = feature_length
        self.classifier_type = classifier_type
        
        # Modulation classes
        self.modulation_classes = [
            'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'FM', 'AM', 'OOK'
        ]
        
        # Initialize classifier
        if SKLEARN_AVAILABLE:
            if classifier_type == 'svm':
                self.classifier = SVC(probability=True)
            elif classifier_type == 'kmeans':
                self.classifier = KMeans(n_clusters=len(self.modulation_classes))
            else:
                self.classifier = None
        else:
            self.classifier = None
        
        # Feature extraction parameters
        self.features_to_extract = [
            'instantaneous_amplitude',
            'instantaneous_phase',
            'instantaneous_frequency',
            'cumulants',
            'spectral_features'
        ]
        
        # Classification results
        self.classification_history = []
        self.current_modulation = 'Unknown'
        
        # Message port for classification results
        self.message_port_register_out(pmt.intern("classification"))
    
    def extract_features(self, signal):
        """
        Extract features for modulation classification
        """
        features = []
        
        # Instantaneous amplitude
        inst_amp = np.abs(signal)
        features.extend([
            np.mean(inst_amp),
            np.std(inst_amp),
            np.max(inst_amp),
            np.min(inst_amp)
        ])
        
        # Instantaneous phase
        inst_phase = np.angle(signal)
        phase_diff = np.diff(np.unwrap(inst_phase))
        features.extend([
            np.mean(phase_diff),
            np.std(phase_diff),
            np.max(phase_diff),
            np.min(phase_diff)
        ])
        
        # Instantaneous frequency
        inst_freq = phase_diff / (2 * np.pi)
        features.extend([
            np.mean(inst_freq),
            np.std(inst_freq),
            np.max(inst_freq),
            np.min(inst_freq)
        ])
        
        # Higher-order statistics (cumulants)
        features.extend(self.calculate_cumulants(signal))
        
        # Spectral features
        fft = np.fft.fft(signal)
        psd = np.abs(fft) ** 2
        features.extend([
            np.max(psd),
            np.argmax(psd),  # Dominant frequency
            np.std(psd),
            self.spectral_entropy(psd)
        ])
        
        # Cyclostationary features
        features.extend(self.cyclostationary_features(signal))
        
        return np.array(features)
    
    def calculate_cumulants(self, signal):
        """Calculate higher-order cumulants"""
        cumulants = []
        
        # Second order (variance)
        c2 = np.var(signal)
        cumulants.append(np.abs(c2))
        
        # Fourth order
        c4 = np.mean(signal ** 4) - 3 * c2 ** 2
        cumulants.append(np.abs(c4))
        
        # Normalized cumulants
        if c2 > 0:
            cumulants.append(np.abs(c4 / (c2 ** 2)))
        else:
            cumulants.append(0)
        
        return cumulants
    
    def spectral_entropy(self, psd):
        """Calculate spectral entropy"""
        psd_norm = psd / np.sum(psd)
        psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
        entropy = -np.sum(psd_norm * np.log2(psd_norm))
        return entropy
    
    def cyclostationary_features(self, signal):
        """Extract cyclostationary features"""
        features = []
        
        # Spectral correlation function (simplified)
        n = len(signal)
        if n > 128:
            # Compute at specific cycle frequencies
            alpha_values = [0, 0.25, 0.5, 1.0]  # Normalized cycle frequencies
            
            for alpha in alpha_values:
                # Simplified spectral correlation
                shift = int(alpha * n)
                if shift < n:
                    correlation = np.mean(signal[:-shift] * np.conj(signal[shift:]))
                    features.append(np.abs(correlation))
                else:
                    features.append(0)
        else:
            features = [0, 0, 0, 0]
        
        return features
    
    def classify_modulation(self, features):
        """
        Classify modulation type based on features
        """
        if self.classifier and hasattr(self.classifier, 'predict'):
            # Use trained classifier
            prediction = self.classifier.predict(features.reshape(1, -1))
            
            if hasattr(self.classifier, 'predict_proba'):
                confidence = np.max(self.classifier.predict_proba(features.reshape(1, -1)))
            else:
                confidence = 1.0
            
            return prediction[0], confidence
        else:
            # Simple rule-based classification
            return self.rule_based_classification(features)
    
    def rule_based_classification(self, features):
        """
        Simple rule-based classification as fallback
        """
        # Extract key features
        amp_std = features[1]  # Amplitude standard deviation
        phase_std = features[5]  # Phase standard deviation
        
        # Simple rules
        if amp_std < 0.1:
            # Constant amplitude suggests PSK
            if phase_std < 0.5:
                return 'BPSK', 0.7
            elif phase_std < 1.0:
                return 'QPSK', 0.6
            else:
                return '8PSK', 0.5
        elif amp_std > 0.5:
            # Variable amplitude suggests QAM or AM
            if phase_std > 1.0:
                return '16QAM', 0.6
            else:
                return 'AM', 0.5
        else:
            return 'Unknown', 0.3
    
    def work(self, input_items, output_items):
        """
        Classify modulation of input signal
        """
        input_signal = input_items[0]
        output_confidence = output_items[0]
        
        # Process in chunks
        num_chunks = len(input_signal) // self.feature_length
        
        for i in range(num_chunks):
            # Extract chunk
            start_idx = i * self.feature_length
            end_idx = start_idx + self.feature_length
            chunk = input_signal[start_idx:end_idx]
            
            # Extract features
            features = self.extract_features(chunk)
            
            # Classify
            modulation, confidence = self.classify_modulation(features)
            
            # Update current modulation
            if isinstance(modulation, (int, np.integer)):
                if modulation < len(self.modulation_classes):
                    self.current_modulation = self.modulation_classes[modulation]
            else:
                self.current_modulation = str(modulation)
            
            # Output confidence
            if i < len(output_confidence):
                output_confidence[i] = confidence
            
            # Store in history
            self.classification_history.append({
                'modulation': self.current_modulation,
                'confidence': confidence,
                'features': features.tolist()
            })
            
            # Publish classification result
            if i % 10 == 0:  # Every 10th classification
                self.publish_classification()
        
        return num_chunks
    
    def publish_classification(self):
        """Publish classification results via message port"""
        if self.classification_history:
            latest = self.classification_history[-1]
            info = {
                'modulation': latest['modulation'],
                'confidence': float(latest['confidence']),
                'history_length': len(self.classification_history)
            }
            
            self.message_port_pub(
                pmt.intern("classification"),
                pmt.to_pmt(info)
            )
    
    def train_classifier(self, training_data, labels):
        """
        Train the classifier with labeled data
        """
        if not SKLEARN_AVAILABLE or not self.classifier:
            print("Classifier training not available")
            return
        
        # Extract features for all training data
        X = []
        for signal in training_data:
            features = self.extract_features(signal)
            X.append(features)
        
        X = np.array(X)
        
        # Train classifier
        self.classifier.fit(X, labels)
        print(f"Classifier trained on {len(X)} samples")


class AnomalyDetector(gr.sync_block):
    """
    Anomaly detection in signal streams
    Uses unsupervised learning to detect unusual patterns
    """
    
    def __init__(self, window_size=1024, sensitivity=2.0):
        """
        Initialize anomaly detector
        
        Args:
            window_size: Size of analysis window
            sensitivity: Sensitivity factor (lower = more sensitive)
        """
        gr.sync_block.__init__(
            self,
            name='Anomaly Detector',
            in_sig=[np.complex64],
            out_sig=[np.float32]  # Anomaly score
        )
        
        self.window_size = window_size
        self.sensitivity = sensitivity
        
        # Statistical model
        self.mean_spectrum = None
        self.std_spectrum = None
        self.samples_seen = 0
        
        # Anomaly detection state
        self.anomaly_scores = []
        self.anomalies_detected = []
        
        # Message port for anomaly alerts
        self.message_port_register_out(pmt.intern("anomaly_alert"))
    
    def update_model(self, spectrum):
        """Update statistical model with new spectrum"""
        if self.mean_spectrum is None:
            self.mean_spectrum = spectrum
            self.std_spectrum = np.zeros_like(spectrum)
        else:
            # Exponential moving average
            alpha = 0.01
            self.mean_spectrum = (1 - alpha) * self.mean_spectrum + alpha * spectrum
            
            # Update standard deviation
            variance = (spectrum - self.mean_spectrum) ** 2
            self.std_spectrum = np.sqrt(
                (1 - alpha) * self.std_spectrum ** 2 + alpha * variance
            )
    
    def calculate_anomaly_score(self, spectrum):
        """Calculate anomaly score for given spectrum"""
        if self.mean_spectrum is None:
            return 0.0
        
        # Z-score based anomaly detection
        z_scores = np.abs((spectrum - self.mean_spectrum) / 
                         (self.std_spectrum + 1e-10))
        
        # Mahalanobis distance (simplified)
        anomaly_score = np.mean(z_scores)
        
        return anomaly_score
    
    def work(self, input_items, output_items):
        """
        Detect anomalies in signal
        """
        input_signal = input_items[0]
        output_scores = output_items[0]
        
        num_windows = len(input_signal) // self.window_size
        
        for i in range(num_windows):
            # Extract window
            start_idx = i * self.window_size
            end_idx = start_idx + self.window_size
            window = input_signal[start_idx:end_idx]
            
            # Compute spectrum
            spectrum = np.abs(np.fft.fft(window)) ** 2
            
            # Calculate anomaly score
            score = self.calculate_anomaly_score(spectrum)
            
            # Update model (only with normal data)
            if score < self.sensitivity:
                self.update_model(spectrum)
            
            # Store score
            self.anomaly_scores.append(score)
            
            # Output score (repeated for window size)
            if i < len(output_scores):
                output_scores[i] = score
            
            # Check for anomaly
            if score > self.sensitivity:
                anomaly_info = {
                    'timestamp': self.samples_seen + start_idx,
                    'score': float(score),
                    'threshold': self.sensitivity
                }
                self.anomalies_detected.append(anomaly_info)
                
                # Send alert
                self.message_port_pub(
                    pmt.intern("anomaly_alert"),
                    pmt.to_pmt(anomaly_info)
                )
                
                print(f"Anomaly detected! Score: {score:.2f}")
            
            self.samples_seen += self.window_size
        
        return num_windows


def test_ml_blocks():
    """Test machine learning blocks"""
    from gnuradio import blocks, analog
    
    print("Testing Machine Learning Blocks...")
    
    class MLTestFlowgraph(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self, "ML Test")
            
            # Generate test signal with modulation
            sample_rate = 32000
            
            # QPSK modulated signal
            data = blocks.vector_source_b(
                np.random.randint(0, 256, 1000).astype(np.uint8),
                repeat=True
            )
            
            # Simple QPSK modulation (placeholder)
            const_source = analog.sig_source_c(
                sample_rate, analog.GR_COS_WAVE, 1000, 1, 0
            )
            
            # Add noise
            noise = analog.noise_source_c(analog.GR_GAUSSIAN, 0.1, 0)
            adder = blocks.add_cc()
            
            # ML blocks
            self.amc = AutomaticModulationClassifier()
            self.anomaly = AnomalyDetector()
            
            # Sinks
            class_sink = blocks.vector_sink_f()
            anomaly_sink = blocks.vector_sink_f()
            
            # Connections
            self.connect(const_source, (adder, 0))
            self.connect(noise, (adder, 1))
            self.connect(adder, self.amc)
            self.connect(adder, self.anomaly)
            self.connect(self.amc, class_sink)
            self.connect(self.anomaly, anomaly_sink)
            
            # Limit runtime
            head = blocks.head(gr.sizeof_gr_complex, 100000)
            self.connect(adder, head)
    
    # Run test
    tb = MLTestFlowgraph()
    tb.run()
    
    print(f"\nClassification: {tb.amc.current_modulation}")
    print(f"Anomalies detected: {len(tb.anomaly.anomalies_detected)}")
    print("\nML blocks test complete!")


if __name__ == '__main__':
    test_ml_blocks()