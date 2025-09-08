#!/usr/bin/env python3
"""
Decimator Block Example - N:1 Input/Output Ratio
Demonstrates creating custom decimation blocks in GNU Radio
Usage: python3 decimator_block.py
"""

import numpy as np
from gnuradio import gr
from gnuradio import blocks
from gnuradio import analog
from gnuradio import audio
from gnuradio.filter import firdes
import matplotlib.pyplot as plt
import time

class PeakPicker(gr.decim_block):
    """
    Peak picking decimator - outputs maximum value in each decimation window
    Useful for envelope detection and peak hold displays
    """
    
    def __init__(self, decimation_factor=10):
        """
        Initialize peak picker
        
        Args:
            decimation_factor: Number of input samples per output sample
        """
        gr.decim_block.__init__(
            self,
            name="Peak Picker",
            in_sig=[np.float32],
            out_sig=[np.float32],
            decim=decimation_factor
        )
        
        self.decimation = decimation_factor
        self.total_input = 0
        self.total_output = 0
        
    def work(self, input_items, output_items):
        """Pick peak from each decimation window"""
        in0 = input_items[0]
        out0 = output_items[0]
        
        # Process each output sample
        for i in range(len(out0)):
            # Get input window for this output sample
            start_idx = i * self.decimation
            end_idx = start_idx + self.decimation
            
            if end_idx <= len(in0):
                # Find peak in window
                window = in0[start_idx:end_idx]
                out0[i] = np.max(np.abs(window))
            else:
                # Not enough input samples
                return 0
        
        self.total_input += len(in0)
        self.total_output += len(out0)
        
        return len(out0)


class MovingAverageDecimator(gr.decim_block):
    """
    Moving average decimator with anti-aliasing
    Averages input samples before decimation to prevent aliasing
    """
    
    def __init__(self, decimation_factor=4, use_filter=True):
        """
        Initialize moving average decimator
        
        Args:
            decimation_factor: Decimation ratio
            use_filter: Apply anti-aliasing filter before averaging
        """
        gr.decim_block.__init__(
            self,
            name="Moving Average Decimator",
            in_sig=[np.float32],
            out_sig=[np.float32],
            decim=decimation_factor
        )
        
        self.decimation = decimation_factor
        self.use_filter = use_filter
        
        # Create anti-aliasing filter taps if needed
        if use_filter:
            # Low-pass filter at Nyquist/decimation
            self.filter_taps = firdes.low_pass(
                1.0,                           # Gain
                1.0,                          # Normalized sample rate
                0.4 / decimation_factor,      # Cutoff (normalized)
                0.1 / decimation_factor,      # Transition width
                firdes.WIN_HAMMING)
            
            # Filter state
            self.filter_state = np.zeros(len(self.filter_taps) - 1)
        
    def work(self, input_items, output_items):
        """Apply filtering and decimation"""
        in0 = input_items[0]
        out0 = output_items[0]
        
        if self.use_filter and len(self.filter_taps) > 0:
            # Apply anti-aliasing filter first
            # Concatenate filter state with input
            filtered_input = np.convolve(
                np.concatenate([self.filter_state, in0]), 
                self.filter_taps, 
                mode='valid')
            
            # Update filter state for next call
            if len(in0) >= len(self.filter_state):
                self.filter_state = in0[-len(self.filter_state):]
            else:
                self.filter_state = np.concatenate([
                    self.filter_state[len(in0):], in0])
        else:
            filtered_input = in0
        
        # Decimate by averaging
        for i in range(len(out0)):
            start_idx = i * self.decimation
            end_idx = start_idx + self.decimation
            
            if end_idx <= len(filtered_input):
                # Average the samples in this window
                out0[i] = np.mean(filtered_input[start_idx:end_idx])
            else:
                return i  # Return number of samples produced
        
        return len(out0)


class MinMaxDecimator(gr.decim_block):
    """
    Outputs both min and max values from each decimation window
    Useful for oscilloscope-style displays
    """
    
    def __init__(self, decimation_factor=10):
        """
        Initialize min-max decimator
        
        Args:
            decimation_factor: Number of input samples per output pair
        """
        gr.decim_block.__init__(
            self,
            name="Min-Max Decimator",
            in_sig=[np.float32],
            out_sig=[np.float32, np.float32],  # Two outputs: min and max
            decim=decimation_factor
        )
        
        self.decimation = decimation_factor
        
    def work(self, input_items, output_items):
        """Find min and max in each window"""
        in0 = input_items[0]
        out_min = output_items[0]
        out_max = output_items[1]
        
        for i in range(len(out_min)):
            start_idx = i * self.decimation
            end_idx = start_idx + self.decimation
            
            if end_idx <= len(in0):
                window = in0[start_idx:end_idx]
                out_min[i] = np.min(window)
                out_max[i] = np.max(window)
            else:
                return i
        
        return len(out_min)


class StatisticalDecimator(gr.decim_block):
    """
    Computes statistical measures over decimation windows
    Outputs mean, std dev, and median
    """
    
    def __init__(self, decimation_factor=100):
        """
        Initialize statistical decimator
        
        Args:
            decimation_factor: Window size for statistics
        """
        gr.decim_block.__init__(
            self,
            name="Statistical Decimator",
            in_sig=[np.float32],
            out_sig=[np.float32, np.float32, np.float32],  # mean, std, median
            decim=decimation_factor
        )
        
        self.decimation = decimation_factor
        self.sample_count = 0
        
    def work(self, input_items, output_items):
        """Compute statistics for each window"""
        in0 = input_items[0]
        out_mean = output_items[0]
        out_std = output_items[1]
        out_median = output_items[2]
        
        for i in range(len(out_mean)):
            start_idx = i * self.decimation
            end_idx = start_idx + self.decimation
            
            if end_idx <= len(in0):
                window = in0[start_idx:end_idx]
                
                # Compute statistics
                out_mean[i] = np.mean(window)
                out_std[i] = np.std(window)
                out_median[i] = np.median(window)
                
                self.sample_count += self.decimation
                
                # Print statistics occasionally
                if self.sample_count % (self.decimation * 100) == 0:
                    print(f"Window {i}: Mean={out_mean[i]:.3f}, "
                          f"Std={out_std[i]:.3f}, Median={out_median[i]:.3f}")
            else:
                return i
        
        return len(out_mean)


class PolyphaseDecimator(gr.decim_block):
    """
    Efficient polyphase decimator implementation
    Splits filter into parallel branches for computational efficiency
    """
    
    def __init__(self, decimation_factor=8, num_taps=64):
        """
        Initialize polyphase decimator
        
        Args:
            decimation_factor: Decimation ratio (must divide num_taps)
            num_taps: Total number of filter taps
        """
        gr.decim_block.__init__(
            self,
            name="Polyphase Decimator",
            in_sig=[np.float32],
            out_sig=[np.float32],
            decim=decimation_factor
        )
        
        self.decimation = decimation_factor
        
        # Design prototype filter
        prototype_filter = firdes.low_pass(
            decimation_factor,               # Gain to compensate for decimation
            1.0,                             # Normalized rate
            0.4 / decimation_factor,         # Cutoff
            0.1 / decimation_factor,         # Transition
            firdes.WIN_HAMMING)
        
        # Ensure filter length is multiple of decimation factor
        filter_len = len(prototype_filter)
        if filter_len % decimation_factor != 0:
            pad_len = decimation_factor - (filter_len % decimation_factor)
            prototype_filter = np.concatenate([
                prototype_filter, np.zeros(pad_len)])
        
        # Split filter into polyphase components
        self.polyphase_taps = []
        for i in range(decimation_factor):
            self.polyphase_taps.append(
                prototype_filter[i::decimation_factor])
        
        # Initialize polyphase filter states
        self.polyphase_states = [
            np.zeros(len(taps)) for taps in self.polyphase_taps]
        
        self.input_buffer = np.array([])
        
    def work(self, input_items, output_items):
        """Apply polyphase filtering and decimation"""
        in0 = input_items[0]
        out0 = output_items[0]
        
        # Add input to buffer
        self.input_buffer = np.concatenate([self.input_buffer, in0])
        
        output_idx = 0
        while len(self.input_buffer) >= self.decimation and output_idx < len(out0):
            # Get decimation block
            block = self.input_buffer[:self.decimation]
            
            # Apply polyphase filtering
            result = 0
            for i in range(self.decimation):
                # Update state for this branch
                self.polyphase_states[i] = np.roll(self.polyphase_states[i], 1)
                self.polyphase_states[i][0] = block[i]
                
                # Compute dot product for this branch
                result += np.dot(self.polyphase_states[i], 
                               self.polyphase_taps[i])
            
            out0[output_idx] = result
            output_idx += 1
            
            # Remove processed samples from buffer
            self.input_buffer = self.input_buffer[self.decimation:]
        
        return output_idx


def create_test_flowgraph():
    """Create test flowgraph for decimator blocks"""
    
    class DecimatorTestFlowgraph(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self, "Decimator Test")
            
            # Parameters
            sample_rate = 48000
            decimation = 10
            
            # Create test signal with multiple frequency components
            self.sig1 = analog.sig_source_f(sample_rate, analog.GR_COS_WAVE, 
                                           100, 0.3, 0)
            self.sig2 = analog.sig_source_f(sample_rate, analog.GR_COS_WAVE, 
                                           1000, 0.3, 0)
            self.sig3 = analog.sig_source_f(sample_rate, analog.GR_COS_WAVE, 
                                           5000, 0.3, 0)
            self.noise = blocks.noise_source_f(blocks.GR_GAUSSIAN, 0.1, 0)
            
            self.adder = blocks.add_vff(1)
            
            # Custom decimator blocks
            self.peak_picker = PeakPicker(decimation)
            self.avg_decimator = MovingAverageDecimator(decimation, True)
            self.minmax_decimator = MinMaxDecimator(decimation)
            self.stats_decimator = StatisticalDecimator(100)
            self.poly_decimator = PolyphaseDecimator(decimation)
            
            # File sinks for analysis
            self.original_sink = blocks.file_sink(
                gr.sizeof_float, '/tmp/original.dat', False)
            self.peak_sink = blocks.file_sink(
                gr.sizeof_float, '/tmp/peaks.dat', False)
            self.avg_sink = blocks.file_sink(
                gr.sizeof_float, '/tmp/averaged.dat', False)
            self.poly_sink = blocks.file_sink(
                gr.sizeof_float, '/tmp/polyphase.dat', False)
            
            # Connections
            self.connect(self.sig1, (self.adder, 0))
            self.connect(self.sig2, (self.adder, 1))
            self.connect(self.sig3, (self.adder, 2))
            self.connect(self.noise, (self.adder, 3))
            
            # Branch to different decimators
            self.connect(self.adder, self.original_sink)
            self.connect(self.adder, self.peak_picker, self.peak_sink)
            self.connect(self.adder, self.avg_decimator, self.avg_sink)
            self.connect(self.adder, self.poly_decimator, self.poly_sink)
            
            # Min-max decimator (two outputs)
            self.connect(self.adder, self.minmax_decimator)
            self.minmax_min_sink = blocks.file_sink(
                gr.sizeof_float, '/tmp/min.dat', False)
            self.minmax_max_sink = blocks.file_sink(
                gr.sizeof_float, '/tmp/max.dat', False)
            self.connect((self.minmax_decimator, 0), self.minmax_min_sink)
            self.connect((self.minmax_decimator, 1), self.minmax_max_sink)
            
            # Statistics decimator (three outputs)
            self.connect(self.adder, self.stats_decimator)
            self.null_sink1 = blocks.null_sink(gr.sizeof_float)
            self.null_sink2 = blocks.null_sink(gr.sizeof_float)
            self.null_sink3 = blocks.null_sink(gr.sizeof_float)
            self.connect((self.stats_decimator, 0), self.null_sink1)
            self.connect((self.stats_decimator, 1), self.null_sink2)
            self.connect((self.stats_decimator, 2), self.null_sink3)
    
    return DecimatorTestFlowgraph()


def analyze_decimation_results(sample_rate=48000, decimation=10, duration=1.0):
    """Analyze and plot decimation results"""
    
    # Read data files
    try:
        original = np.fromfile('/tmp/original.dat', dtype=np.float32)
        peaks = np.fromfile('/tmp/peaks.dat', dtype=np.float32)
        averaged = np.fromfile('/tmp/averaged.dat', dtype=np.float32)
        polyphase = np.fromfile('/tmp/polyphase.dat', dtype=np.float32)
        min_vals = np.fromfile('/tmp/min.dat', dtype=np.float32)
        max_vals = np.fromfile('/tmp/max.dat', dtype=np.float32)
    except FileNotFoundError:
        print("Data files not found. Run the decimator test first.")
        return
    
    # Limit data for plotting
    plot_samples = min(1000, len(original))
    decimated_samples = plot_samples // decimation
    
    # Create time axes
    t_original = np.arange(plot_samples) / sample_rate
    t_decimated = np.arange(decimated_samples) * decimation / sample_rate
    
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Original signal
    axes[0, 0].plot(t_original, original[:plot_samples], 'b-', alpha=0.7)
    axes[0, 0].set_title('Original Signal')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True)
    
    # Peak picker
    axes[0, 1].plot(t_original, original[:plot_samples], 'b-', alpha=0.3)
    axes[0, 1].plot(t_decimated, peaks[:decimated_samples], 'ro-', 
                   markersize=4, label='Peaks')
    axes[0, 1].set_title('Peak Picker Decimator')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Moving average
    axes[1, 0].plot(t_original, original[:plot_samples], 'b-', alpha=0.3)
    axes[1, 0].plot(t_decimated, averaged[:decimated_samples], 'g.-', 
                   label='Averaged')
    axes[1, 0].set_title('Moving Average Decimator (with anti-aliasing)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Polyphase
    axes[1, 1].plot(t_original, original[:plot_samples], 'b-', alpha=0.3)
    axes[1, 1].plot(t_decimated, polyphase[:decimated_samples], 'm.-', 
                   label='Polyphase')
    axes[1, 1].set_title('Polyphase Decimator')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Min-Max envelope
    axes[2, 0].plot(t_original, original[:plot_samples], 'b-', alpha=0.3)
    axes[2, 0].fill_between(t_decimated, 
                           min_vals[:decimated_samples],
                           max_vals[:decimated_samples],
                           alpha=0.5, color='orange', label='Min-Max Envelope')
    axes[2, 0].set_title('Min-Max Decimator')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Amplitude')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    # Frequency comparison
    from scipy import signal
    f_orig, psd_orig = signal.welch(original, sample_rate, nperseg=1024)
    f_dec, psd_avg = signal.welch(averaged, sample_rate/decimation, nperseg=128)
    f_dec_poly, psd_poly = signal.welch(polyphase, sample_rate/decimation, nperseg=128)
    
    axes[2, 1].semilogy(f_orig, psd_orig, 'b-', alpha=0.5, label='Original')
    axes[2, 1].semilogy(f_dec, psd_avg, 'g-', label='Moving Avg')
    axes[2, 1].semilogy(f_dec_poly, psd_poly, 'm-', label='Polyphase')
    axes[2, 1].set_title('Frequency Response Comparison')
    axes[2, 1].set_xlabel('Frequency (Hz)')
    axes[2, 1].set_ylabel('PSD')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to demonstrate decimator blocks"""
    
    print("GNU Radio Custom Decimator Block Examples")
    print("=" * 50)
    print("\nThis example demonstrates various decimation techniques:")
    print("1. Peak Picker - Preserves peak values")
    print("2. Moving Average - Anti-aliased decimation")
    print("3. Min-Max - Preserves signal envelope")
    print("4. Statistical - Computes statistics over windows")
    print("5. Polyphase - Efficient multi-rate filtering")
    print("\nTest signal: 100Hz + 1kHz + 5kHz + noise")
    print(f"Decimation factor: 10")
    print("=" * 50)
    
    # Create and run flowgraph
    tb = create_test_flowgraph()
    
    print("\nRunning decimation test for 2 seconds...")
    tb.start()
    time.sleep(2)
    tb.stop()
    tb.wait()
    
    print("\nDecimation complete!")
    print("\nAnalyzing results...")
    
    # Analyze and plot results
    analyze_decimation_results()
    
    print("\nDone!")


if __name__ == '__main__':
    main()