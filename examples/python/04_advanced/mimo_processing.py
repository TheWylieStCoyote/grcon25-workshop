#!/usr/bin/env python3
"""
MIMO Processing and Beamforming
Advanced example showing Multiple-Input Multiple-Output techniques
"""

import numpy as np
from gnuradio import gr
import pmt

class MIMOBeamformer(gr.sync_block):
    """
    MIMO Beamformer implementing various beamforming algorithms
    Supports MRC, EGC, Zero-Forcing, and MMSE
    """
    
    def __init__(self, num_rx_antennas=4, num_tx_antennas=2, 
                 algorithm='mrc', steering_vector=None):
        """
        Initialize MIMO beamformer
        
        Args:
            num_rx_antennas: Number of receive antennas
            num_tx_antennas: Number of transmit antennas
            algorithm: Beamforming algorithm ('mrc', 'egc', 'zf', 'mmse')
            steering_vector: Optional steering vector for beam direction
        """
        # Multiple inputs for different antennas
        in_sig = [np.complex64] * num_rx_antennas
        # Multiple outputs for spatial streams
        out_sig = [np.complex64] * num_tx_antennas
        
        gr.sync_block.__init__(
            self,
            name='MIMO Beamformer',
            in_sig=in_sig,
            out_sig=out_sig
        )
        
        self.num_rx = num_rx_antennas
        self.num_tx = num_tx_antennas
        self.algorithm = algorithm
        
        # Beamforming weights
        self.weights = np.eye(num_rx_antennas, num_tx_antennas, dtype=np.complex64)
        
        # Channel matrix estimate
        self.H = np.eye(num_rx_antennas, num_tx_antennas, dtype=np.complex64)
        
        # Steering vector for directional beamforming
        if steering_vector is None:
            self.steering_vector = np.ones(num_rx_antennas, dtype=np.complex64)
        else:
            self.steering_vector = np.array(steering_vector, dtype=np.complex64)
        
        # Noise variance for MMSE
        self.noise_variance = 0.01
        
        # Performance metrics
        self.condition_numbers = []
        self.capacity_estimates = []
        
        # Message ports
        self.message_port_register_in(pmt.intern("channel_est"))
        self.message_port_register_out(pmt.intern("beam_info"))
        self.set_msg_handler(pmt.intern("channel_est"), self.handle_channel_estimate)
    
    def work(self, input_items, output_items):
        """
        Apply MIMO beamforming
        """
        num_samples = len(input_items[0])
        
        # Stack inputs into matrix (antennas x samples)
        X = np.zeros((self.num_rx, num_samples), dtype=np.complex64)
        for i in range(self.num_rx):
            X[i, :] = input_items[i]
        
        # Update beamforming weights based on algorithm
        self.update_weights()
        
        # Apply beamforming: Y = W^H * X
        Y = np.dot(self.weights.conj().T, X)
        
        # Output spatial streams
        for i in range(self.num_tx):
            if i < Y.shape[0]:
                output_items[i][:] = Y[i, :]
            else:
                output_items[i][:] = 0
        
        # Calculate and publish performance metrics periodically
        if np.random.rand() < 0.01:  # 1% of the time
            self.calculate_metrics()
            self.publish_beam_info()
        
        return num_samples
    
    def update_weights(self):
        """Update beamforming weights based on selected algorithm"""
        
        if self.algorithm == 'mrc':
            # Maximum Ratio Combining
            self.weights = self.H / np.linalg.norm(self.H, axis=0, keepdims=True)
            
        elif self.algorithm == 'egc':
            # Equal Gain Combining
            self.weights = np.exp(1j * np.angle(self.H))
            self.weights /= np.sqrt(self.num_rx)
            
        elif self.algorithm == 'zf':
            # Zero-Forcing (requires H to be full rank)
            try:
                # W = H * (H^H * H)^-1
                gram = np.dot(self.H.conj().T, self.H)
                if np.linalg.matrix_rank(gram) == gram.shape[0]:
                    self.weights = np.dot(self.H, np.linalg.inv(gram))
                else:
                    # Fall back to MRC if singular
                    self.weights = self.H / np.linalg.norm(self.H, axis=0, keepdims=True)
            except np.linalg.LinAlgError:
                self.weights = self.H
                
        elif self.algorithm == 'mmse':
            # Minimum Mean Square Error
            try:
                # W = H * (H^H * H + noise * I)^-1
                gram = np.dot(self.H.conj().T, self.H)
                regularized = gram + self.noise_variance * np.eye(self.num_tx)
                self.weights = np.dot(self.H, np.linalg.inv(regularized))
            except np.linalg.LinAlgError:
                self.weights = self.H
        
        # Apply steering vector for directional beamforming
        self.weights = self.weights * self.steering_vector[:, np.newaxis]
    
    def handle_channel_estimate(self, msg):
        """Handle channel matrix estimate from external estimator"""
        try:
            channel_data = pmt.to_python(msg)
            if isinstance(channel_data, dict) and 'H' in channel_data:
                H_flat = np.array(channel_data['H'])
                self.H = H_flat.reshape((self.num_rx, self.num_tx))
                print(f"Updated channel matrix, condition number: "
                      f"{np.linalg.cond(self.H):.2f}")
        except Exception as e:
            print(f"Error updating channel estimate: {e}")
    
    def calculate_metrics(self):
        """Calculate MIMO performance metrics"""
        # Condition number (lower is better)
        cond = np.linalg.cond(self.H)
        self.condition_numbers.append(cond)
        
        # Channel capacity (Shannon capacity for MIMO)
        # C = log2(det(I + SNR/Nt * H * H^H))
        snr = 1.0 / self.noise_variance
        identity = np.eye(self.num_rx)
        capacity_matrix = identity + (snr / self.num_tx) * np.dot(self.H, self.H.conj().T)
        capacity = np.log2(np.abs(np.linalg.det(capacity_matrix)))
        self.capacity_estimates.append(capacity)
    
    def publish_beam_info(self):
        """Publish beamforming information"""
        info = {
            'algorithm': self.algorithm,
            'condition_number': self.condition_numbers[-1] if self.condition_numbers else 0,
            'capacity_bps_hz': self.capacity_estimates[-1] if self.capacity_estimates else 0,
            'weights_norm': np.linalg.norm(self.weights),
            'effective_channels': np.linalg.matrix_rank(self.H)
        }
        
        self.message_port_pub(
            pmt.intern("beam_info"),
            pmt.to_pmt(info)
        )


class MIMOChannelEstimator(gr.sync_block):
    """
    MIMO Channel Estimator using training sequences
    Implements LS and MMSE channel estimation
    """
    
    def __init__(self, num_rx=4, num_tx=2, training_length=64, 
                 estimation_method='ls'):
        """
        Initialize MIMO channel estimator
        
        Args:
            num_rx: Number of receive antennas
            num_tx: Number of transmit antennas  
            training_length: Length of training sequence
            estimation_method: 'ls' or 'mmse'
        """
        gr.sync_block.__init__(
            self,
            name='MIMO Channel Estimator',
            in_sig=[np.complex64] * num_rx,
            out_sig=[np.complex64] * num_rx  # Pass through with channel info
        )
        
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.training_length = training_length
        self.method = estimation_method
        
        # Generate orthogonal training sequences (Hadamard matrix based)
        self.training_sequences = self.generate_training_sequences()
        
        # Channel estimate
        self.H = np.eye(num_rx, num_tx, dtype=np.complex64)
        
        # Tracking
        self.sample_count = 0
        self.estimation_period = 1000  # Estimate every N samples
        
        # Message port for channel estimates
        self.message_port_register_out(pmt.intern("channel_est"))
    
    def generate_training_sequences(self):
        """Generate orthogonal training sequences for each TX antenna"""
        # Use Hadamard matrix for orthogonality
        def hadamard(n):
            if n == 1:
                return np.array([[1]])
            else:
                h_n_1 = hadamard(n // 2)
                top = np.hstack([h_n_1, h_n_1])
                bottom = np.hstack([h_n_1, -h_n_1])
                return np.vstack([top, bottom])
        
        # Find next power of 2
        n = 2 ** int(np.ceil(np.log2(max(self.num_tx, self.training_length))))
        H_matrix = hadamard(n)
        
        # Extract training sequences
        sequences = H_matrix[:self.num_tx, :self.training_length]
        
        # Convert to complex and normalize
        sequences = sequences.astype(np.complex64)
        sequences /= np.sqrt(self.training_length)
        
        return sequences
    
    def work(self, input_items, output_items):
        """
        Estimate MIMO channel from training sequences
        """
        num_samples = len(input_items[0])
        
        # Pass through inputs
        for i in range(self.num_rx):
            output_items[i][:] = input_items[i]
        
        # Check if it's time to estimate channel
        if self.sample_count % self.estimation_period == 0:
            # Collect received training signals
            Y = np.zeros((self.num_rx, self.training_length), dtype=np.complex64)
            
            for i in range(self.num_rx):
                if len(input_items[i]) >= self.training_length:
                    Y[i, :] = input_items[i][:self.training_length]
            
            # Estimate channel
            if self.method == 'ls':
                self.estimate_channel_ls(Y)
            else:  # mmse
                self.estimate_channel_mmse(Y)
            
            # Publish channel estimate
            self.publish_channel_estimate()
        
        self.sample_count += num_samples
        return num_samples
    
    def estimate_channel_ls(self, Y):
        """
        Least Squares channel estimation
        Y = H * X + N
        H_est = Y * X^H * (X * X^H)^-1
        """
        X = self.training_sequences
        
        try:
            # Calculate LS estimate
            X_pseudo_inv = np.linalg.pinv(X.T)
            self.H = np.dot(Y, X_pseudo_inv.T)
        except np.linalg.LinAlgError:
            print("LS estimation failed, keeping previous estimate")
    
    def estimate_channel_mmse(self, Y):
        """
        Minimum Mean Square Error channel estimation
        Includes noise statistics
        """
        X = self.training_sequences
        
        # Estimate noise variance from received signal
        noise_var = self.estimate_noise_variance(Y)
        
        try:
            # MMSE: H_est = Y * X^H * (X * X^H + noise * I)^-1
            gram = np.dot(X, X.conj().T)
            regularized = gram + noise_var * np.eye(self.num_tx)
            inv_reg = np.linalg.inv(regularized)
            self.H = np.dot(np.dot(Y, X.conj().T), inv_reg)
        except np.linalg.LinAlgError:
            # Fall back to LS
            self.estimate_channel_ls(Y)
    
    def estimate_noise_variance(self, Y):
        """Estimate noise variance from received signal"""
        # Simple estimator: use variance of later samples
        if Y.shape[1] > self.training_length:
            noise_samples = Y[:, self.training_length:]
            return np.var(noise_samples)
        else:
            return 0.01  # Default
    
    def publish_channel_estimate(self):
        """Publish channel matrix estimate"""
        channel_dict = {
            'H': self.H.flatten().tolist(),
            'shape': self.H.shape,
            'method': self.method,
            'rank': np.linalg.matrix_rank(self.H),
            'condition': np.linalg.cond(self.H)
        }
        
        self.message_port_pub(
            pmt.intern("channel_est"),
            pmt.to_pmt(channel_dict)
        )


class SpatialMultiplexer(gr.sync_block):
    """
    Spatial multiplexing for MIMO systems
    Implements V-BLAST style encoding/decoding
    """
    
    def __init__(self, num_streams=2, modulation='qpsk'):
        """
        Initialize spatial multiplexer
        
        Args:
            num_streams: Number of spatial streams
            modulation: Modulation type ('bpsk', 'qpsk', 'qam16', 'qam64')
        """
        gr.sync_block.__init__(
            self,
            name='Spatial Multiplexer',
            in_sig=[np.uint8],  # Byte input
            out_sig=[np.complex64] * num_streams  # Multiple spatial streams
        )
        
        self.num_streams = num_streams
        self.modulation = modulation
        
        # Modulation parameters
        self.bits_per_symbol = {
            'bpsk': 1, 'qpsk': 2, 'qam16': 4, 'qam64': 6
        }[modulation]
        
        # Symbol mapping
        self.constellation = self.create_constellation()
        
        # Interleaver for spatial diversity
        self.interleaver_size = 1024
        self.interleaver_pattern = np.random.permutation(self.interleaver_size)
        
    def create_constellation(self):
        """Create constellation points for modulation"""
        if self.modulation == 'bpsk':
            return np.array([-1, 1], dtype=np.complex64)
        elif self.modulation == 'qpsk':
            return np.array([
                1+1j, 1-1j, -1+1j, -1-1j
            ], dtype=np.complex64) / np.sqrt(2)
        elif self.modulation == 'qam16':
            # 16-QAM constellation
            points = []
            for i in [-3, -1, 1, 3]:
                for q in [-3, -1, 1, 3]:
                    points.append(i + 1j*q)
            return np.array(points, dtype=np.complex64) / np.sqrt(10)
        else:  # qam64
            # 64-QAM constellation
            points = []
            for i in [-7, -5, -3, -1, 1, 3, 5, 7]:
                for q in [-7, -5, -3, -1, 1, 3, 5, 7]:
                    points.append(i + 1j*q)
            return np.array(points, dtype=np.complex64) / np.sqrt(42)
    
    def work(self, input_items, output_items):
        """
        Spatially multiplex input data
        """
        input_bytes = input_items[0]
        num_bytes = len(input_bytes)
        
        # Convert bytes to bits
        bits = np.unpackbits(input_bytes.astype(np.uint8))
        
        # Calculate number of symbols
        num_symbols = len(bits) // (self.bits_per_symbol * self.num_streams)
        used_bits = num_symbols * self.bits_per_symbol * self.num_streams
        
        if num_symbols == 0:
            return 0
        
        # Reshape bits for each stream
        bits = bits[:used_bits].reshape(self.num_streams, -1, self.bits_per_symbol)
        
        # Map to symbols for each stream
        for stream in range(self.num_streams):
            symbols = np.zeros(num_symbols, dtype=np.complex64)
            
            for i in range(num_symbols):
                # Convert bits to symbol index
                symbol_bits = bits[stream, i, :]
                symbol_idx = 0
                for j, bit in enumerate(symbol_bits):
                    symbol_idx += bit * (2 ** j)
                
                # Map to constellation point
                if symbol_idx < len(self.constellation):
                    symbols[i] = self.constellation[symbol_idx]
            
            # Apply interleaving for diversity
            if num_symbols <= self.interleaver_size:
                interleaved = np.zeros_like(symbols)
                pattern = self.interleaver_pattern[:num_symbols]
                interleaved[pattern] = symbols
                symbols = interleaved
            
            # Output to spatial stream
            output_items[stream][:num_symbols] = symbols
        
        return num_symbols


def test_mimo_system():
    """Test MIMO processing chain"""
    from gnuradio import blocks
    import matplotlib.pyplot as plt
    
    print("Testing MIMO System...")
    
    class MIMOTestFlowgraph(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self, "MIMO Test")
            
            # Parameters
            num_rx = 4
            num_tx = 2
            num_samples = 10000
            
            # Data source
            data_source = blocks.vector_source_b(
                np.random.randint(0, 256, 1000).astype(np.uint8), 
                repeat=True
            )
            
            # Spatial multiplexer
            spatial_mux = SpatialMultiplexer(num_streams=num_tx, modulation='qpsk')
            
            # Simulate MIMO channel (simplified)
            channel_taps = np.random.randn(num_rx, num_tx) + \
                          1j * np.random.randn(num_rx, num_tx)
            channel_taps /= np.sqrt(2)
            
            # Create channel mixing
            channel_blocks = []
            adders = []
            for rx in range(num_rx):
                adder = blocks.add_cc()
                adders.append(adder)
                for tx in range(num_tx):
                    mult = blocks.multiply_const_cc(channel_taps[rx, tx])
                    channel_blocks.append(mult)
                    self.connect((spatial_mux, tx), mult, (adder, tx))
            
            # Channel estimator
            estimator = MIMOChannelEstimator(
                num_rx=num_rx, num_tx=num_tx, 
                estimation_method='mmse'
            )
            
            # Beamformer
            beamformer = MIMOBeamformer(
                num_rx_antennas=num_rx, 
                num_tx_antennas=num_tx,
                algorithm='mmse'
            )
            
            # Connect channel outputs to estimator and beamformer
            for rx in range(num_rx):
                self.connect(adders[rx], (estimator, rx))
                self.connect((estimator, rx), (beamformer, rx))
            
            # Output sinks
            self.sinks = []
            for tx in range(num_tx):
                sink = blocks.vector_sink_c()
                self.sinks.append(sink)
                self.connect((beamformer, tx), sink)
            
            # Connect message passing
            self.msg_connect(estimator, "channel_est", beamformer, "channel_est")
            
            # Data source
            self.connect(data_source, spatial_mux)
            
            # Limit samples
            head = blocks.head(gr.sizeof_char, 1000)
            self.connect(data_source, head)
    
    # Run test
    tb = MIMOTestFlowgraph()
    tb.run()
    
    print("\nMIMO system test complete!")
    print("Check output streams for spatial multiplexing results")


if __name__ == '__main__':
    test_mimo_system()