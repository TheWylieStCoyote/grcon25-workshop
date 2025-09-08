#!/usr/bin/env python3
"""
Adaptive Filter using LMS Algorithm
Advanced example showing real-time adaptive filtering
"""

import numpy as np
from gnuradio import gr
import pmt

class AdaptiveLMSFilter(gr.sync_block):
    """
    Adaptive filter using Least Mean Squares (LMS) algorithm
    Useful for channel equalization, echo cancellation, and interference removal
    """
    
    def __init__(self, num_taps=32, mu=0.01, training_symbols=None):
        """
        Initialize adaptive LMS filter
        
        Args:
            num_taps: Number of filter taps
            mu: Step size (learning rate)
            training_symbols: Known training sequence for adaptation
        """
        gr.sync_block.__init__(
            self,
            name='Adaptive LMS Filter',
            in_sig=[np.complex64, np.complex64],  # Signal and reference
            out_sig=[np.complex64, np.float32]     # Filtered output and error
        )
        
        # Filter parameters
        self.num_taps = num_taps
        self.mu = mu
        
        # Filter state
        self.weights = np.zeros(num_taps, dtype=np.complex64)
        self.weights[num_taps//2] = 1.0  # Initialize as delay
        self.buffer = np.zeros(num_taps, dtype=np.complex64)
        
        # Training mode
        self.training_symbols = training_symbols
        self.training_index = 0
        self.training_mode = training_symbols is not None
        
        # Performance metrics
        self.convergence_history = []
        self.mse_history = []
        self.iteration = 0
        
        # Message ports for control
        self.message_port_register_in(pmt.intern("control"))
        self.message_port_register_out(pmt.intern("taps"))
        self.set_msg_handler(pmt.intern("control"), self.handle_control)
        
    def work(self, input_items, output_items):
        """
        Process samples using LMS adaptation
        """
        signal = input_items[0]
        reference = input_items[1]
        output = output_items[0]
        error_out = output_items[1]
        
        num_samples = len(signal)
        
        for i in range(num_samples):
            # Shift input into buffer
            self.buffer = np.roll(self.buffer, 1)
            self.buffer[0] = signal[i]
            
            # Apply filter
            y = np.dot(self.weights.conj(), self.buffer)
            output[i] = y
            
            # Calculate error
            if self.training_mode and self.training_index < len(self.training_symbols):
                # Use training symbols
                desired = self.training_symbols[self.training_index]
                self.training_index += 1
            else:
                # Use reference signal
                desired = reference[i]
                self.training_mode = False
            
            error = desired - y
            error_out[i] = np.abs(error)
            
            # LMS weight update
            self.weights += self.mu * error.conj() * self.buffer
            
            # Normalize weights to prevent divergence
            weight_norm = np.linalg.norm(self.weights)
            if weight_norm > 10.0:
                self.weights /= weight_norm
            
            # Track convergence
            self.iteration += 1
            if self.iteration % 100 == 0:
                mse = np.abs(error) ** 2
                self.mse_history.append(mse)
                self.convergence_history.append(np.copy(self.weights))
                
                # Publish filter taps via message
                if self.iteration % 1000 == 0:
                    self.publish_taps()
        
        return num_samples
    
    def handle_control(self, msg):
        """Handle control messages"""
        try:
            control = pmt.to_python(msg)
            
            if isinstance(control, dict):
                if 'reset' in control and control['reset']:
                    self.reset_filter()
                
                if 'mu' in control:
                    self.mu = float(control['mu'])
                    print(f"Updated learning rate to {self.mu}")
                
                if 'training' in control:
                    self.training_symbols = np.array(control['training'])
                    self.training_index = 0
                    self.training_mode = True
                    print(f"Loaded {len(self.training_symbols)} training symbols")
                    
        except Exception as e:
            print(f"Error handling control message: {e}")
    
    def publish_taps(self):
        """Publish current filter taps via message port"""
        taps_dict = {
            'taps': self.weights.tolist(),
            'iteration': self.iteration,
            'mse': self.mse_history[-1] if self.mse_history else 0
        }
        self.message_port_pub(
            pmt.intern("taps"),
            pmt.to_pmt(taps_dict)
        )
    
    def reset_filter(self):
        """Reset filter to initial state"""
        self.weights = np.zeros(self.num_taps, dtype=np.complex64)
        self.weights[self.num_taps//2] = 1.0
        self.buffer = np.zeros(self.num_taps, dtype=np.complex64)
        self.convergence_history = []
        self.mse_history = []
        self.iteration = 0
        print("Filter reset")
    
    def get_convergence_info(self):
        """Get convergence statistics"""
        if not self.mse_history:
            return None
        
        return {
            'iterations': self.iteration,
            'final_mse': self.mse_history[-1],
            'min_mse': np.min(self.mse_history),
            'convergence_rate': len(self.mse_history),
            'final_weights': self.weights.tolist()
        }


class ChannelEstimator(gr.sync_block):
    """
    Advanced channel estimation using pilot symbols
    Implements MMSE channel estimation with interpolation
    """
    
    def __init__(self, pilot_carriers, pilot_symbols, fft_size=64):
        """
        Initialize channel estimator
        
        Args:
            pilot_carriers: Indices of pilot subcarriers
            pilot_symbols: Known pilot symbol values
            fft_size: FFT size for OFDM
        """
        gr.sync_block.__init__(
            self,
            name='Channel Estimator',
            in_sig=[np.complex64],
            out_sig=[np.complex64, np.complex64]  # Equalized signal, channel estimate
        )
        
        self.pilot_carriers = np.array(pilot_carriers)
        self.pilot_symbols = np.array(pilot_symbols)
        self.fft_size = fft_size
        
        # Channel estimation state
        self.channel_estimate = np.ones(fft_size, dtype=np.complex64)
        self.noise_variance = 0.01
        
        # Smoothing filter for channel estimates
        self.alpha = 0.1  # Smoothing factor
        
    def work(self, input_items, output_items):
        """
        Estimate and equalize channel
        """
        input_signal = input_items[0]
        output_equalized = output_items[0]
        output_channel = output_items[1]
        
        # Process in OFDM symbol chunks
        num_symbols = len(input_signal) // self.fft_size
        
        for sym_idx in range(num_symbols):
            # Extract OFDM symbol
            symbol_start = sym_idx * self.fft_size
            symbol_end = symbol_start + self.fft_size
            ofdm_symbol = input_signal[symbol_start:symbol_end]
            
            # FFT to frequency domain
            freq_domain = np.fft.fft(ofdm_symbol)
            
            # Extract pilots and estimate channel at pilot positions
            received_pilots = freq_domain[self.pilot_carriers]
            channel_at_pilots = received_pilots / self.pilot_symbols
            
            # Interpolate channel estimate across all subcarriers
            channel_estimate_new = self.interpolate_channel(channel_at_pilots)
            
            # Smooth channel estimate
            self.channel_estimate = (1 - self.alpha) * self.channel_estimate + \
                                   self.alpha * channel_estimate_new
            
            # MMSE equalization
            channel_power = np.abs(self.channel_estimate) ** 2
            mmse_equalizer = self.channel_estimate.conj() / \
                           (channel_power + self.noise_variance)
            
            # Equalize
            equalized = freq_domain * mmse_equalizer
            
            # IFFT back to time domain
            time_domain = np.fft.ifft(equalized)
            
            # Output
            output_equalized[symbol_start:symbol_end] = time_domain
            output_channel[symbol_start:symbol_end] = np.fft.ifft(self.channel_estimate)
        
        return num_symbols * self.fft_size
    
    def interpolate_channel(self, channel_at_pilots):
        """
        Interpolate channel estimate from pilots to all subcarriers
        Using cubic spline interpolation
        """
        from scipy import interpolate
        
        # Create interpolation function
        # Handle wrap-around for cyclic prefix
        extended_pilots = np.concatenate([
            self.pilot_carriers - self.fft_size,
            self.pilot_carriers,
            self.pilot_carriers + self.fft_size
        ])
        extended_values = np.tile(channel_at_pilots, 3)
        
        # Interpolate real and imaginary parts separately
        interp_real = interpolate.interp1d(
            extended_pilots,
            extended_values.real,
            kind='cubic',
            fill_value='extrapolate'
        )
        interp_imag = interpolate.interp1d(
            extended_pilots,
            extended_values.imag,
            kind='cubic',
            fill_value='extrapolate'
        )
        
        # Evaluate at all subcarrier positions
        all_carriers = np.arange(self.fft_size)
        channel_estimate = interp_real(all_carriers) + 1j * interp_imag(all_carriers)
        
        return channel_estimate


class CognitiveSpectrumSensor(gr.sync_block):
    """
    Advanced spectrum sensing for cognitive radio
    Implements energy detection with adaptive thresholding
    """
    
    def __init__(self, fft_size=1024, averaging_length=100):
        gr.sync_block.__init__(
            self,
            name='Cognitive Spectrum Sensor',
            in_sig=[np.complex64],
            out_sig=[np.float32]  # Spectrum occupancy vector
        )
        
        self.fft_size = fft_size
        self.averaging_length = averaging_length
        
        # Detection parameters
        self.false_alarm_rate = 0.01
        self.noise_floor_estimate = np.ones(fft_size) * -100  # dBm
        self.threshold_factor = 3.0  # dB above noise floor
        
        # Averaging buffer
        self.spectrum_buffer = np.zeros((averaging_length, fft_size))
        self.buffer_index = 0
        
        # Output message port for detailed spectrum info
        self.message_port_register_out(pmt.intern("spectrum_info"))
        
    def work(self, input_items, output_items):
        """
        Perform spectrum sensing
        """
        input_signal = input_items[0]
        occupancy = output_items[0]
        
        # Process in FFT-sized chunks
        num_ffts = len(input_signal) // self.fft_size
        
        for i in range(num_ffts):
            # Extract chunk and compute PSD
            chunk = input_signal[i*self.fft_size:(i+1)*self.fft_size]
            fft_result = np.fft.fftshift(np.fft.fft(chunk))
            psd = 10 * np.log10(np.abs(fft_result) ** 2 + 1e-10)
            
            # Update averaging buffer
            self.spectrum_buffer[self.buffer_index] = psd
            self.buffer_index = (self.buffer_index + 1) % self.averaging_length
            
            # Compute averaged spectrum
            avg_spectrum = np.mean(self.spectrum_buffer, axis=0)
            
            # Adaptive noise floor estimation (minimum statistics)
            self.update_noise_floor(avg_spectrum)
            
            # Detect occupied channels
            threshold = self.noise_floor_estimate + self.threshold_factor
            occupied = (avg_spectrum > threshold).astype(np.float32)
            
            # Output occupancy vector
            start_idx = i * self.fft_size
            end_idx = start_idx + self.fft_size
            if end_idx <= len(occupancy):
                occupancy[start_idx:end_idx] = np.repeat(
                    np.mean(occupied), self.fft_size
                )
            
            # Periodically publish detailed spectrum info
            if i % 10 == 0:
                self.publish_spectrum_info(avg_spectrum, occupied)
        
        return num_ffts * self.fft_size
    
    def update_noise_floor(self, spectrum):
        """
        Update noise floor estimate using minimum statistics
        """
        # Simple exponential averaging of minimum values
        alpha = 0.01
        self.noise_floor_estimate = (1 - alpha) * self.noise_floor_estimate + \
                                   alpha * np.minimum(spectrum, self.noise_floor_estimate + 10)
    
    def publish_spectrum_info(self, spectrum, occupancy):
        """
        Publish detailed spectrum information
        """
        # Find occupied bands
        occupied_indices = np.where(occupancy)[0]
        
        if len(occupied_indices) > 0:
            # Group consecutive indices into bands
            bands = []
            start = occupied_indices[0]
            end = start
            
            for idx in occupied_indices[1:]:
                if idx == end + 1:
                    end = idx
                else:
                    bands.append((start, end))
                    start = idx
                    end = idx
            bands.append((start, end))
            
            info = {
                'occupied_bands': bands,
                'occupancy_percent': np.mean(occupancy) * 100,
                'peak_power': np.max(spectrum),
                'noise_floor': np.mean(self.noise_floor_estimate)
            }
            
            self.message_port_pub(
                pmt.intern("spectrum_info"),
                pmt.to_pmt(info)
            )


def test_adaptive_filter():
    """Test the adaptive filter with a simple channel"""
    from gnuradio import blocks, analog
    import matplotlib.pyplot as plt
    
    print("Testing Adaptive LMS Filter...")
    
    # Create test flowgraph
    class TestFlowgraph(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self, "Adaptive Filter Test")
            
            # Parameters
            sample_rate = 32000
            num_samples = 10000
            
            # Create test signal
            signal_source = analog.sig_source_c(
                sample_rate, analog.GR_COS_WAVE, 1000, 1, 0
            )
            
            # Add multipath channel (simplified)
            delay = blocks.delay(gr.sizeof_gr_complex, 10)
            multiply = blocks.multiply_const_cc(0.5 * np.exp(1j * np.pi/4))
            adder = blocks.add_cc()
            
            # Add noise
            noise = analog.noise_source_c(analog.GR_GAUSSIAN, 0.01, 0)
            adder2 = blocks.add_cc()
            
            # Adaptive filter
            self.lms_filter = AdaptiveLMSFilter(num_taps=32, mu=0.01)
            
            # Reference signal (delayed original)
            reference_delay = blocks.delay(gr.sizeof_gr_complex, 15)
            
            # Sinks
            output_sink = blocks.vector_sink_c()
            error_sink = blocks.vector_sink_f()
            
            # Connections - create multipath
            self.connect(signal_source, (adder, 0))
            self.connect(signal_source, delay, multiply, (adder, 1))
            self.connect(adder, (adder2, 0))
            self.connect(noise, (adder2, 1))
            
            # Connect to adaptive filter
            self.connect(adder2, (self.lms_filter, 0))
            self.connect(signal_source, reference_delay, (self.lms_filter, 1))
            
            # Connect outputs
            self.connect((self.lms_filter, 0), output_sink)
            self.connect((self.lms_filter, 1), error_sink)
            
            # Head to limit samples
            head = blocks.head(gr.sizeof_gr_complex, num_samples)
            self.connect(signal_source, head)
    
    # Run test
    tb = TestFlowgraph()
    tb.run()
    
    # Get convergence info
    info = tb.lms_filter.get_convergence_info()
    if info:
        print(f"\nAdaptive Filter Results:")
        print(f"  Iterations: {info['iterations']}")
        print(f"  Final MSE: {info['final_mse']:.6f}")
        print(f"  Minimum MSE: {info['min_mse']:.6f}")
        print(f"  Convergence rate: {info['convergence_rate']} measurements")
    
    print("\nAdaptive filter test complete!")


if __name__ == '__main__':
    test_adaptive_filter()