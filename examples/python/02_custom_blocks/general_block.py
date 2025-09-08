#!/usr/bin/env python3
"""
General Block Example - Variable Input/Output Ratios
Demonstrates creating custom general blocks with flexible I/O in GNU Radio
Usage: python3 general_block.py
"""

import numpy as np
from gnuradio import gr
from gnuradio import blocks
from gnuradio import analog
import time
import struct

class PacketDetector(gr.general_block):
    """
    Packet detector with variable output based on sync word detection
    Demonstrates general_block with variable input/output consumption
    """
    
    def __init__(self, sync_word=0xABCD, packet_len=100):
        """
        Initialize packet detector
        
        Args:
            sync_word: 16-bit sync pattern to detect
            packet_len: Expected packet length after sync
        """
        gr.general_block.__init__(
            self,
            name="Packet Detector",
            in_sig=[np.uint8],
            out_sig=[np.uint8]
        )
        
        self.sync_word = sync_word
        self.packet_len = packet_len
        self.state = 'SEARCHING'
        self.buffer = []
        self.packet_buffer = []
        self.sync_buffer = []
        
        # Convert sync word to bytes
        self.sync_bytes = [(sync_word >> 8) & 0xFF, sync_word & 0xFF]
        
        # Statistics
        self.packets_found = 0
        self.false_syncs = 0
        
    def forecast(self, noutput_items, ninput_items_required):
        """
        Tell scheduler how many input items we need
        This is called by scheduler before work()
        """
        # We need at least 2 bytes to check for sync word
        # and potentially a full packet
        for i in range(len(ninput_items_required)):
            ninput_items_required[i] = max(2, self.packet_len + 2)
    
    def general_work(self, input_items, output_items):
        """
        Process input and detect packets
        Returns number of items consumed/produced
        """
        in0 = input_items[0]
        out0 = output_items[0]
        
        consumed = 0
        produced = 0
        
        while consumed < len(in0) - 1 and produced < len(out0):
            if self.state == 'SEARCHING':
                # Look for sync word
                if (in0[consumed] == self.sync_bytes[0] and 
                    in0[consumed + 1] == self.sync_bytes[1]):
                    # Sync word found!
                    self.state = 'COLLECTING'
                    self.packet_buffer = []
                    consumed += 2
                    self.packets_found += 1
                    print(f"Sync word detected! Packet #{self.packets_found}")
                else:
                    consumed += 1
                    
            elif self.state == 'COLLECTING':
                # Collect packet data
                remaining_packet = self.packet_len - len(self.packet_buffer)
                remaining_input = len(in0) - consumed
                remaining_output = len(out0) - produced
                
                to_copy = min(remaining_packet, remaining_input, remaining_output)
                
                # Copy data to output
                for i in range(to_copy):
                    out0[produced] = in0[consumed]
                    self.packet_buffer.append(in0[consumed])
                    consumed += 1
                    produced += 1
                
                # Check if packet complete
                if len(self.packet_buffer) >= self.packet_len:
                    self.state = 'SEARCHING'
                    print(f"Packet complete: {self.packet_len} bytes")
        
        self.consume(0, consumed)
        self.produce(0, produced)
        
        return gr.WORK_CALLED_PRODUCE


class VariableRateResampler(gr.general_block):
    """
    Resampler with dynamically variable rate
    Demonstrates general_block with changing I/O ratios
    """
    
    def __init__(self, initial_rate=1.0):
        """
        Initialize variable rate resampler
        
        Args:
            initial_rate: Initial resampling rate
        """
        gr.general_block.__init__(
            self,
            name="Variable Rate Resampler",
            in_sig=[np.float32],
            out_sig=[np.float32]
        )
        
        self.rate = initial_rate
        self.phase = 0.0
        self.history = np.zeros(4)  # For interpolation
        self.set_history(4)  # Tell GNU Radio we need history
        
    def forecast(self, noutput_items, ninput_items_required):
        """Calculate required input items based on rate"""
        # We need at least rate * output items as input
        for i in range(len(ninput_items_required)):
            ninput_items_required[i] = int(np.ceil(noutput_items * self.rate)) + 4
    
    def linear_interpolate(self, samples, fraction):
        """Linear interpolation between samples"""
        idx = int(fraction)
        frac = fraction - idx
        
        if idx + 1 < len(samples):
            return samples[idx] * (1 - frac) + samples[idx + 1] * frac
        else:
            return samples[idx]
    
    def general_work(self, input_items, output_items):
        """Apply variable rate resampling"""
        in0 = input_items[0]
        out0 = output_items[0]
        
        consumed = 0
        produced = 0
        
        while produced < len(out0) and self.phase < len(in0) - 1:
            # Interpolate at current phase
            out0[produced] = self.linear_interpolate(in0, self.phase)
            produced += 1
            
            # Advance phase by rate
            self.phase += self.rate
        
        # Calculate how many samples we consumed
        consumed = int(self.phase)
        self.phase -= consumed
        
        self.consume(0, consumed)
        self.produce(0, produced)
        
        return gr.WORK_CALLED_PRODUCE
    
    def set_rate(self, rate):
        """Update resampling rate"""
        self.rate = max(0.1, min(10.0, rate))
        print(f"Resampling rate set to {self.rate}")


class BurstTagger(gr.general_block):
    """
    Tags bursts of energy in the signal
    Demonstrates stream tags and burst detection
    """
    
    def __init__(self, threshold=0.1, min_burst_len=10):
        """
        Initialize burst tagger
        
        Args:
            threshold: Energy threshold for burst detection
            min_burst_len: Minimum samples for valid burst
        """
        gr.general_block.__init__(
            self,
            name="Burst Tagger",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        
        self.threshold = threshold
        self.min_burst_len = min_burst_len
        self.in_burst = False
        self.burst_start = 0
        self.burst_count = 0
        self.current_burst_len = 0
        
    def forecast(self, noutput_items, ninput_items_required):
        """Need same number of inputs as outputs"""
        for i in range(len(ninput_items_required)):
            ninput_items_required[i] = noutput_items
    
    def general_work(self, input_items, output_items):
        """Detect and tag bursts"""
        in0 = input_items[0]
        out0 = output_items[0]
        
        # Pass through signal
        n = min(len(in0), len(out0))
        out0[:n] = in0[:n]
        
        # Detect bursts
        for i in range(n):
            energy = np.abs(in0[i])**2
            
            if not self.in_burst and energy > self.threshold:
                # Burst start
                self.in_burst = True
                self.burst_start = self.nitems_written(0) + i
                self.current_burst_len = 1
                
                # Add start tag
                key = gr.pmt.intern("burst_start")
                value = gr.pmt.from_long(self.burst_count)
                self.add_item_tag(0, self.burst_start, key, value)
                
            elif self.in_burst and energy > self.threshold:
                # Continue burst
                self.current_burst_len += 1
                
            elif self.in_burst and energy <= self.threshold:
                # Burst end
                if self.current_burst_len >= self.min_burst_len:
                    # Valid burst
                    burst_end = self.nitems_written(0) + i
                    
                    # Add end tag
                    key = gr.pmt.intern("burst_end")
                    value = gr.pmt.from_long(self.burst_count)
                    self.add_item_tag(0, burst_end, key, value)
                    
                    # Add length tag
                    key = gr.pmt.intern("burst_length")
                    value = gr.pmt.from_long(self.current_burst_len)
                    self.add_item_tag(0, burst_end, key, value)
                    
                    print(f"Burst {self.burst_count}: {self.current_burst_len} samples")
                    self.burst_count += 1
                    
                self.in_burst = False
                self.current_burst_len = 0
        
        self.consume(0, n)
        self.produce(0, n)
        
        return gr.WORK_CALLED_PRODUCE


class AdaptiveFilter(gr.general_block):
    """
    Adaptive filter using LMS algorithm
    Demonstrates general_block with adaptive processing
    """
    
    def __init__(self, num_taps=32, mu=0.01):
        """
        Initialize adaptive filter
        
        Args:
            num_taps: Number of filter taps
            mu: LMS step size
        """
        gr.general_block.__init__(
            self,
            name="Adaptive Filter",
            in_sig=[np.complex64, np.complex64],  # Input and reference
            out_sig=[np.complex64, np.float32]     # Output and error
        )
        
        self.num_taps = num_taps
        self.mu = mu
        self.taps = np.zeros(num_taps, dtype=complex)
        self.taps[num_taps//2] = 1.0  # Initialize as delay
        self.buffer = np.zeros(num_taps, dtype=complex)
        
    def forecast(self, noutput_items, ninput_items_required):
        """Need equal inputs on both channels"""
        for i in range(len(ninput_items_required)):
            ninput_items_required[i] = noutput_items + self.num_taps
    
    def general_work(self, input_items, output_items):
        """Apply adaptive filtering with LMS"""
        signal = input_items[0]
        reference = input_items[1]
        out_signal = output_items[0]
        out_error = output_items[1]
        
        n = min(len(signal), len(reference), len(out_signal), len(out_error))
        
        for i in range(n):
            # Update buffer
            self.buffer = np.roll(self.buffer, -1)
            self.buffer[-1] = signal[i]
            
            # Apply filter
            filtered = np.dot(self.taps, self.buffer)
            out_signal[i] = filtered
            
            # Calculate error
            error = reference[i] - filtered
            out_error[i] = np.abs(error)**2
            
            # LMS update
            self.taps += self.mu * np.conj(error) * self.buffer
        
        self.consume(0, n)
        self.consume(1, n)
        self.produce(0, n)
        self.produce(1, n)
        
        return gr.WORK_CALLED_PRODUCE


class CorrelationDetector(gr.general_block):
    """
    Correlates input with known pattern and outputs correlation peaks
    Demonstrates pattern matching and selective output
    """
    
    def __init__(self, pattern, threshold=0.8):
        """
        Initialize correlation detector
        
        Args:
            pattern: Known pattern to correlate with
            threshold: Correlation threshold for detection
        """
        gr.general_block.__init__(
            self,
            name="Correlation Detector",
            in_sig=[np.complex64],
            out_sig=[np.complex64, np.float32]  # Signal and correlation
        )
        
        self.pattern = np.array(pattern, dtype=complex)
        self.pattern_conj = np.conj(self.pattern[::-1])  # For correlation
        self.threshold = threshold
        self.pattern_len = len(pattern)
        self.buffer = np.zeros(self.pattern_len * 2, dtype=complex)
        self.detections = 0
        
    def forecast(self, noutput_items, ninput_items_required):
        """Need enough input for correlation"""
        for i in range(len(ninput_items_required)):
            ninput_items_required[i] = noutput_items + self.pattern_len
    
    def general_work(self, input_items, output_items):
        """Correlate and detect patterns"""
        in0 = input_items[0]
        out_signal = output_items[0]
        out_corr = output_items[1]
        
        n = min(len(in0) - self.pattern_len, len(out_signal), len(out_corr))
        
        for i in range(n):
            # Get window
            window = in0[i:i+self.pattern_len]
            
            # Compute correlation
            correlation = np.abs(np.dot(window, self.pattern_conj))
            normalized_corr = correlation / (np.linalg.norm(window) * 
                                            np.linalg.norm(self.pattern) + 1e-10)
            
            # Output signal and correlation
            out_signal[i] = in0[i]
            out_corr[i] = normalized_corr
            
            # Check for detection
            if normalized_corr > self.threshold:
                self.detections += 1
                
                # Add detection tag
                key = gr.pmt.intern("pattern_detected")
                value = gr.pmt.from_double(normalized_corr)
                self.add_item_tag(0, self.nitems_written(0) + i, key, value)
                
                print(f"Pattern detected! Correlation: {normalized_corr:.3f}, "
                      f"Total: {self.detections}")
        
        self.consume(0, n)
        self.produce(0, n)
        self.produce(1, n)
        
        return gr.WORK_CALLED_PRODUCE


def create_test_flowgraph():
    """Create test flowgraph for general blocks"""
    
    class GeneralBlockTestFlowgraph(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self, "General Block Test")
            
            # Create test signal with bursts
            sample_rate = 10000
            
            # Burst signal source
            self.burst_source = analog.sig_source_c(
                sample_rate, analog.GR_COS_WAVE, 1000, 0.5, 0)
            
            # Gate for creating bursts
            self.gate_control = analog.sig_source_f(
                sample_rate, analog.GR_SQR_WAVE, 2, 1, 0)
            self.gate_threshold = blocks.threshold_ff(0.5, 0.5, 0)
            self.float_to_complex = blocks.float_to_complex()
            self.multiply = blocks.multiply_cc()
            
            # Custom blocks
            self.burst_tagger = BurstTagger(threshold=0.1, min_burst_len=100)
            self.var_resampler = VariableRateResampler(initial_rate=0.5)
            
            # Create packet test source
            self.packet_source = blocks.vector_source_b(
                [0x00] * 50 + [0xAB, 0xCD] + list(range(100)) + [0x00] * 50,
                True, 1)
            self.packet_detector = PacketDetector(sync_word=0xABCD, packet_len=100)
            
            # Sinks
            self.null_sink1 = blocks.null_sink(gr.sizeof_gr_complex)
            self.null_sink2 = blocks.null_sink(gr.sizeof_char)
            self.null_sink3 = blocks.null_sink(gr.sizeof_float)
            
            # Connections for burst detection
            self.connect(self.gate_control, self.gate_threshold)
            self.connect(self.gate_threshold, (self.float_to_complex, 0))
            self.connect(blocks.null_source(gr.sizeof_float), 
                        (self.float_to_complex, 1))
            self.connect(self.burst_source, (self.multiply, 0))
            self.connect(self.float_to_complex, (self.multiply, 1))
            self.connect(self.multiply, self.burst_tagger)
            self.connect(self.burst_tagger, self.null_sink1)
            
            # Packet detection chain
            self.connect(self.packet_source, self.packet_detector)
            self.connect(self.packet_detector, self.null_sink2)
            
            # Variable rate resampler (using burst source)
            self.audio_source = analog.sig_source_f(
                sample_rate, analog.GR_COS_WAVE, 440, 1.0, 0)
            self.connect(self.audio_source, self.var_resampler)
            self.connect(self.var_resampler, self.null_sink3)
            
    return GeneralBlockTestFlowgraph()


def main():
    """Main function to demonstrate general blocks"""
    
    print("GNU Radio General Block Examples")
    print("=" * 50)
    print("\nThis example demonstrates general blocks with:")
    print("1. Packet Detector - Variable I/O based on sync detection")
    print("2. Burst Tagger - Adds stream tags for burst detection")
    print("3. Variable Rate Resampler - Dynamic resampling rates")
    print("4. Adaptive Filter - LMS adaptive filtering")
    print("5. Correlation Detector - Pattern matching")
    print("=" * 50)
    
    # Create and run test flowgraph
    tb = create_test_flowgraph()
    
    print("\nStarting general block test...")
    print("You should see:")
    print("- Packet sync detections")
    print("- Burst detections with lengths")
    print("- Variable rate resampling updates")
    
    tb.start()
    
    # Run for a while with rate changes
    time.sleep(2)
    
    print("\nChanging resampling rate to 2.0...")
    tb.var_resampler.set_rate(2.0)
    time.sleep(2)
    
    print("\nChanging resampling rate to 0.25...")
    tb.var_resampler.set_rate(0.25)
    time.sleep(2)
    
    print("\nStopping...")
    tb.stop()
    tb.wait()
    
    print("\nGeneral block test complete!")


if __name__ == '__main__':
    main()