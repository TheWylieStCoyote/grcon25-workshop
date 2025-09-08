#!/usr/bin/env python3
"""
Sync Block Example - 1:1 Input/Output Ratio
Demonstrates creating custom synchronous blocks in GNU Radio
Usage: python3 sync_block_example.py
"""

import numpy as np
from gnuradio import gr
from gnuradio import blocks
from gnuradio import analog
from gnuradio import audio
import time
import sys

class AmplitudeModulator(gr.sync_block):
    """
    Custom AM modulator block with adjustable modulation index
    Demonstrates sync_block with 1:1 input/output ratio
    """
    
    def __init__(self, modulation_index=0.5, carrier_freq=1000, sample_rate=48000):
        """
        Initialize AM modulator
        
        Args:
            modulation_index: Modulation depth (0 to 1)
            carrier_freq: Carrier frequency in Hz
            sample_rate: Sample rate in Hz
        """
        gr.sync_block.__init__(
            self,
            name="AM Modulator",
            in_sig=[np.float32],  # One float input
            out_sig=[np.float32]  # One float output
        )
        
        self.modulation_index = modulation_index
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        self.phase = 0
        self.phase_increment = 2 * np.pi * carrier_freq / sample_rate
        
    def work(self, input_items, output_items):
        """
        Process input samples and generate output
        This is called by the GNU Radio scheduler
        """
        in0 = input_items[0]
        out0 = output_items[0]
        
        # Generate carrier and modulate
        for i in range(len(in0)):
            # Generate carrier
            carrier = np.cos(self.phase)
            
            # Apply AM modulation: output = (1 + m * input) * carrier
            out0[i] = (1 + self.modulation_index * in0[i]) * carrier
            
            # Update phase
            self.phase += self.phase_increment
            if self.phase >= 2 * np.pi:
                self.phase -= 2 * np.pi
        
        # Return number of samples processed
        return len(output_items[0])
    
    def set_modulation_index(self, modulation_index):
        """Setter for modulation index"""
        self.modulation_index = max(0, min(1, modulation_index))
    
    def set_carrier_freq(self, freq):
        """Setter for carrier frequency"""
        self.carrier_freq = freq
        self.phase_increment = 2 * np.pi * freq / self.sample_rate


class SignalClipper(gr.sync_block):
    """
    Custom signal clipper/limiter block
    Demonstrates sync_block with configurable parameters
    """
    
    def __init__(self, clip_level=0.8, soft_clip=False):
        """
        Initialize signal clipper
        
        Args:
            clip_level: Maximum amplitude (0 to 1)
            soft_clip: Use soft clipping (tanh) instead of hard clipping
        """
        gr.sync_block.__init__(
            self,
            name="Signal Clipper",
            in_sig=[np.float32],
            out_sig=[np.float32]
        )
        
        self.clip_level = clip_level
        self.soft_clip = soft_clip
        
    def work(self, input_items, output_items):
        """Process samples with clipping"""
        in0 = input_items[0]
        out0 = output_items[0]
        
        if self.soft_clip:
            # Soft clipping using tanh
            out0[:] = self.clip_level * np.tanh(in0 / self.clip_level)
        else:
            # Hard clipping
            out0[:] = np.clip(in0, -self.clip_level, self.clip_level)
        
        return len(output_items[0])
    
    def set_clip_level(self, level):
        """Update clip level"""
        self.clip_level = max(0.001, min(1.0, level))


class RunningStatistics(gr.sync_block):
    """
    Calculate running statistics of input signal
    Demonstrates sync_block with internal state
    """
    
    def __init__(self, window_size=1000, print_interval=1.0):
        """
        Initialize statistics calculator
        
        Args:
            window_size: Number of samples for statistics window
            print_interval: How often to print statistics (seconds)
        """
        gr.sync_block.__init__(
            self,
            name="Running Statistics",
            in_sig=[np.float32],
            out_sig=[np.float32]  # Pass-through
        )
        
        self.window_size = window_size
        self.print_interval = print_interval
        self.buffer = np.zeros(window_size)
        self.buffer_index = 0
        self.last_print_time = time.time()
        self.total_samples = 0
        
        # Statistics
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.sum_val = 0
        self.sum_squared = 0
        
    def work(self, input_items, output_items):
        """Calculate statistics and pass through signal"""
        in0 = input_items[0]
        out0 = output_items[0]
        
        # Pass through
        out0[:] = in0
        
        # Update statistics
        for sample in in0:
            # Update buffer
            old_sample = self.buffer[self.buffer_index]
            self.buffer[self.buffer_index] = sample
            self.buffer_index = (self.buffer_index + 1) % self.window_size
            
            # Update running statistics
            self.sum_val += sample - old_sample
            self.sum_squared += sample**2 - old_sample**2
            self.total_samples += 1
            
            # Track min/max
            if sample < self.min_val:
                self.min_val = sample
            if sample > self.max_val:
                self.max_val = sample
        
        # Print statistics periodically
        current_time = time.time()
        if current_time - self.last_print_time >= self.print_interval:
            self.print_statistics()
            self.last_print_time = current_time
        
        return len(output_items[0])
    
    def print_statistics(self):
        """Print current statistics"""
        if self.total_samples < self.window_size:
            n = self.total_samples
        else:
            n = self.window_size
        
        if n > 0:
            mean = self.sum_val / n
            variance = (self.sum_squared / n) - mean**2
            std_dev = np.sqrt(max(0, variance))
            
            print(f"\n--- Signal Statistics (last {n} samples) ---")
            print(f"Mean:     {mean:.6f}")
            print(f"Std Dev:  {std_dev:.6f}")
            print(f"Min:      {self.min_val:.6f}")
            print(f"Max:      {self.max_val:.6f}")
            print(f"Range:    {self.max_val - self.min_val:.6f}")
            print(f"RMS:      {np.sqrt(self.sum_squared/n):.6f}")


class BiQuadFilter(gr.sync_block):
    """
    Biquad IIR filter implementation
    Demonstrates sync_block with filter state
    """
    
    def __init__(self, filter_type='lowpass', frequency=1000, 
                 sample_rate=48000, q_factor=0.707):
        """
        Initialize biquad filter
        
        Args:
            filter_type: 'lowpass', 'highpass', 'bandpass', 'notch'
            frequency: Cutoff/center frequency
            sample_rate: Sample rate
            q_factor: Q factor (resonance)
        """
        gr.sync_block.__init__(
            self,
            name="BiQuad Filter",
            in_sig=[np.float32],
            out_sig=[np.float32]
        )
        
        self.filter_type = filter_type
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.q_factor = q_factor
        
        # Filter state variables
        self.x1 = 0  # x[n-1]
        self.x2 = 0  # x[n-2]
        self.y1 = 0  # y[n-1]
        self.y2 = 0  # y[n-2]
        
        # Calculate filter coefficients
        self.update_coefficients()
    
    def update_coefficients(self):
        """Calculate biquad filter coefficients"""
        omega = 2 * np.pi * self.frequency / self.sample_rate
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        alpha = sin_omega / (2 * self.q_factor)
        
        if self.filter_type == 'lowpass':
            b0 = (1 - cos_omega) / 2
            b1 = 1 - cos_omega
            b2 = (1 - cos_omega) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_omega
            a2 = 1 - alpha
            
        elif self.filter_type == 'highpass':
            b0 = (1 + cos_omega) / 2
            b1 = -(1 + cos_omega)
            b2 = (1 + cos_omega) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_omega
            a2 = 1 - alpha
            
        elif self.filter_type == 'bandpass':
            b0 = alpha
            b1 = 0
            b2 = -alpha
            a0 = 1 + alpha
            a1 = -2 * cos_omega
            a2 = 1 - alpha
            
        elif self.filter_type == 'notch':
            b0 = 1
            b1 = -2 * cos_omega
            b2 = 1
            a0 = 1 + alpha
            a1 = -2 * cos_omega
            a2 = 1 - alpha
            
        else:
            # Default to pass-through
            b0, b1, b2 = 1, 0, 0
            a0, a1, a2 = 1, 0, 0
        
        # Normalize coefficients
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0
    
    def work(self, input_items, output_items):
        """Apply biquad filter"""
        in0 = input_items[0]
        out0 = output_items[0]
        
        for i in range(len(in0)):
            # Direct Form I implementation
            x0 = in0[i]
            y0 = self.b0 * x0 + self.b1 * self.x1 + self.b2 * self.x2
            y0 -= self.a1 * self.y1 + self.a2 * self.y2
            
            # Update state
            self.x2 = self.x1
            self.x1 = x0
            self.y2 = self.y1
            self.y1 = y0
            
            out0[i] = y0
        
        return len(output_items[0])
    
    def set_frequency(self, freq):
        """Update filter frequency"""
        self.frequency = freq
        self.update_coefficients()
    
    def set_q_factor(self, q):
        """Update Q factor"""
        self.q_factor = max(0.1, q)
        self.update_coefficients()


def create_test_flowgraph():
    """Create a flowgraph to test custom blocks"""
    
    class TestFlowgraph(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self, "Custom Block Test")
            
            # Parameters
            sample_rate = 48000
            
            # Signal sources
            self.audio_source = analog.sig_source_f(
                sample_rate, analog.GR_COS_WAVE, 440, 0.3, 0)
            
            # Custom blocks
            self.am_modulator = AmplitudeModulator(
                modulation_index=0.8, 
                carrier_freq=5000, 
                sample_rate=sample_rate)
            
            self.clipper = SignalClipper(
                clip_level=0.9, 
                soft_clip=True)
            
            self.statistics = RunningStatistics(
                window_size=1000,
                print_interval=2.0)
            
            self.biquad = BiQuadFilter(
                filter_type='bandpass',
                frequency=5000,
                sample_rate=sample_rate,
                q_factor=5.0)
            
            # Output
            self.audio_sink = audio.sink(sample_rate, '', True)
            
            # Connections
            self.connect(self.audio_source, self.am_modulator)
            self.connect(self.am_modulator, self.biquad)
            self.connect(self.biquad, self.clipper)
            self.connect(self.clipper, self.statistics)
            self.connect(self.statistics, self.audio_sink)
    
    return TestFlowgraph()


def main():
    """Main function to demonstrate custom sync blocks"""
    
    print("GNU Radio Custom Sync Block Examples")
    print("=" * 50)
    print("\nThis example demonstrates several custom sync blocks:")
    print("1. AM Modulator - Modulates audio signal onto carrier")
    print("2. BiQuad Filter - Bandpass filter around carrier")
    print("3. Signal Clipper - Soft clipping for limiting")
    print("4. Running Statistics - Real-time signal analysis")
    print("\nThe signal chain is:")
    print("440Hz tone -> AM Modulator (5kHz carrier) -> Bandpass Filter")
    print("-> Soft Clipper -> Statistics Monitor -> Audio Output")
    print("\nYou should hear a 5kHz carrier modulated by 440Hz")
    print("\nPress Ctrl+C to stop")
    print("=" * 50)
    
    # Create and run flowgraph
    tb = create_test_flowgraph()
    tb.start()
    
    try:
        # Run for a while
        time.sleep(10)
        
        print("\n\nChanging modulation index to 0.3...")
        tb.am_modulator.set_modulation_index(0.3)
        time.sleep(5)
        
        print("Changing carrier frequency to 3kHz...")
        tb.am_modulator.set_carrier_freq(3000)
        tb.biquad.set_frequency(3000)
        time.sleep(5)
        
        print("Adjusting filter Q factor...")
        tb.biquad.set_q_factor(10.0)
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    
    tb.stop()
    tb.wait()
    print("Done!")


if __name__ == '__main__':
    main()