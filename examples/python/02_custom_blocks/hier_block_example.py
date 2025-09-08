#!/usr/bin/env python3
"""
Hierarchical Block Example - Composing Complex Blocks
Demonstrates creating reusable hierarchical blocks in GNU Radio
Usage: python3 hier_block_example.py
"""

import numpy as np
from gnuradio import gr
from gnuradio import blocks
from gnuradio import analog
from gnuradio import filter as gr_filter
from gnuradio.filter import firdes
from gnuradio import audio
import time

class ChannelModel(gr.hier_block2):
    """
    Hierarchical block implementing a complete channel model
    Includes noise, frequency offset, timing offset, and multipath
    """
    
    def __init__(self, noise_voltage=0.1, frequency_offset=0.01, 
                 time_offset=1.0, taps=[1.0, 0, 0.3]):
        """
        Initialize channel model
        
        Args:
            noise_voltage: Noise amplitude
            frequency_offset: Normalized frequency offset
            time_offset: Timing offset factor
            taps: Multipath channel taps
        """
        gr.hier_block2.__init__(
            self,
            "Channel Model",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),  # Input signature
            gr.io_signature(1, 1, gr.sizeof_gr_complex)   # Output signature
        )
        
        # Create internal blocks
        self.timing_offset = gr_filter.mmse_resampler_cc(0, time_offset)
        self.multipath = gr_filter.fir_filter_ccc(1, taps)
        self.frequency_offset = blocks.rotator_cc(2 * np.pi * frequency_offset)
        self.noise_adder = blocks.add_cc()
        self.noise = analog.noise_source_c(analog.GR_GAUSSIAN, noise_voltage, 0)
        
        # Connect internal blocks
        self.connect(self, self.timing_offset)
        self.connect(self.timing_offset, self.multipath)
        self.connect(self.multipath, self.frequency_offset)
        self.connect(self.frequency_offset, (self.noise_adder, 0))
        self.connect(self.noise, (self.noise_adder, 1))
        self.connect(self.noise_adder, self)
        
        # Store parameters for adjustment
        self.noise_voltage = noise_voltage
        self.freq_offset = frequency_offset
        self.time_off = time_offset
        
    def set_noise_voltage(self, noise_voltage):
        """Update noise level"""
        self.noise_voltage = noise_voltage
        self.noise.set_amplitude(noise_voltage)
    
    def set_frequency_offset(self, freq_offset):
        """Update frequency offset"""
        self.freq_offset = freq_offset
        self.frequency_offset.set_phase_inc(2 * np.pi * freq_offset)
    
    def set_timing_offset(self, time_offset):
        """Update timing offset"""
        self.time_off = time_offset
        self.timing_offset.set_resamp_ratio(time_offset)


class AGC(gr.hier_block2):
    """
    Automatic Gain Control hierarchical block
    Combines multiple processing stages for robust AGC
    """
    
    def __init__(self, rate=1e-4, reference=1.0, gain=1.0, max_gain=100.0):
        """
        Initialize AGC
        
        Args:
            rate: AGC update rate
            reference: Target output level
            gain: Initial gain
            max_gain: Maximum allowed gain
        """
        gr.hier_block2.__init__(
            self,
            "AGC",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(1, 1, gr.sizeof_gr_complex)
        )
        
        # Create AGC chain
        self.agc = analog.agc2_cc(rate, reference, gain, max_gain)
        
        # Add optional components
        self.input_power_squelch = analog.pwr_squelch_cc(-50, 1e-4, 0, False)
        self.output_limiter = blocks.multiply_const_cc(1.0)
        
        # Connect
        self.connect(self, self.input_power_squelch)
        self.connect(self.input_power_squelch, self.agc)
        self.connect(self.agc, self.output_limiter)
        self.connect(self.output_limiter, self)
        
    def set_rate(self, rate):
        """Update AGC rate"""
        self.agc.set_rate(rate)
    
    def set_reference(self, reference):
        """Update reference level"""
        self.agc.set_reference(reference)
    
    def set_gain(self, gain):
        """Update initial gain"""
        self.agc.set_gain(gain)
    
    def set_max_gain(self, max_gain):
        """Update maximum gain"""
        self.agc.set_max_gain(max_gain)


class FMTransceiver(gr.hier_block2):
    """
    Complete FM transceiver as hierarchical block
    Includes modulator and demodulator
    """
    
    def __init__(self, audio_rate=48000, if_rate=480000, deviation=5000):
        """
        Initialize FM transceiver
        
        Args:
            audio_rate: Audio sample rate
            if_rate: Intermediate frequency rate
            deviation: FM deviation in Hz
        """
        gr.hier_block2.__init__(
            self,
            "FM Transceiver",
            gr.io_signature(1, 1, gr.sizeof_float),     # Audio input
            gr.io_signature(2, 2, gr.sizeof_float)      # Audio output, IF output
        )
        
        self.audio_rate = audio_rate
        self.if_rate = if_rate
        self.deviation = deviation
        
        # Transmit chain
        self.audio_filter_tx = gr_filter.fir_filter_fff(
            1, firdes.low_pass(1, audio_rate, 3000, 500))
        
        self.interpolator = gr_filter.rational_resampler_fff(
            interpolation=int(if_rate/audio_rate),
            decimation=1,
            taps=None)
        
        self.fm_mod = analog.frequency_modulator_fc(
            2 * np.pi * deviation / if_rate)
        
        # Receive chain
        self.fm_demod = analog.quadrature_demod_cf(
            if_rate / (2 * np.pi * deviation))
        
        self.decimator = gr_filter.rational_resampler_fff(
            interpolation=1,
            decimation=int(if_rate/audio_rate),
            taps=None)
        
        self.audio_filter_rx = gr_filter.fir_filter_fff(
            1, firdes.low_pass(1, audio_rate, 3000, 500))
        
        # Create loopback for testing
        self.connect(self, self.audio_filter_tx)
        self.connect(self.audio_filter_tx, self.interpolator)
        self.connect(self.interpolator, self.fm_mod)
        
        # Split to IF output and demod
        self.connect(self.fm_mod, self.fm_demod)
        self.connect(self.fm_demod, self.decimator)
        self.connect(self.decimator, self.audio_filter_rx)
        self.connect(self.audio_filter_rx, (self, 0))  # Audio output
        
        # IF output (convert complex to real)
        self.complex_to_real = blocks.complex_to_real()
        self.connect(self.fm_mod, self.complex_to_real)
        self.connect(self.complex_to_real, (self, 1))  # IF output


class DigitalPLL(gr.hier_block2):
    """
    Digital Phase-Locked Loop hierarchical block
    For carrier recovery and synchronization
    """
    
    def __init__(self, loop_bw=0.01, max_freq=0.1, min_freq=-0.1):
        """
        Initialize digital PLL
        
        Args:
            loop_bw: Loop bandwidth
            max_freq: Maximum frequency deviation (normalized)
            min_freq: Minimum frequency deviation (normalized)
        """
        gr.hier_block2.__init__(
            self,
            "Digital PLL",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(3, 3, gr.sizeof_gr_complex)  # Signal, VCO, Error
        )
        
        # Create PLL
        self.pll = analog.pll_carriertracking_cc(
            loop_bw, max_freq, min_freq)
        
        # Get error signal
        self.conjugate = blocks.conjugate_cc()
        self.multiply = blocks.multiply_cc()
        self.complex_to_arg = blocks.complex_to_arg()
        self.float_to_complex = blocks.float_to_complex()
        
        # Connect PLL
        self.connect(self, self.pll)
        self.connect(self.pll, (self, 0))  # Locked signal output
        
        # Generate VCO output for monitoring
        self.vco = blocks.vco_c(1.0, 1.0, 1.0)
        self.connect(self.pll, self.complex_to_arg)
        self.connect(self.complex_to_arg, self.vco)
        self.connect(self.vco, (self, 1))  # VCO output
        
        # Calculate error
        self.connect(self, (self.multiply, 0))
        self.connect(self.pll, self.conjugate)
        self.connect(self.conjugate, (self.multiply, 1))
        self.connect(self.multiply, (self, 2))  # Error output


class SpectrumSensor(gr.hier_block2):
    """
    Spectrum sensing hierarchical block
    Performs FFT and energy detection
    """
    
    def __init__(self, fft_size=1024, sample_rate=1e6, 
                 center_freq=0, threshold=-60):
        """
        Initialize spectrum sensor
        
        Args:
            fft_size: FFT size
            sample_rate: Sample rate
            center_freq: Center frequency
            threshold: Detection threshold in dB
        """
        gr.hier_block2.__init__(
            self,
            "Spectrum Sensor",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(2, 2, gr.sizeof_float * fft_size)  # Spectrum, Detections
        )
        
        self.fft_size = fft_size
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.threshold = threshold
        
        # Create FFT chain
        from gnuradio import fft
        from gnuradio.fft import window
        
        self.stream_to_vector = blocks.stream_to_vector(
            gr.sizeof_gr_complex, fft_size)
        
        self.fft = fft.fft_vcc(
            fft_size, True, window.blackmanharris(fft_size), True, 1)
        
        self.complex_to_mag_squared = blocks.complex_to_mag_squared(fft_size)
        
        self.log10 = blocks.nlog10_ff(10, fft_size, -20)
        
        # Single pole IIR filter for averaging
        self.avg = blocks.single_pole_iir_filter_ff(0.9, fft_size)
        
        # Threshold detector
        self.threshold_detector = blocks.threshold_ff(
            threshold, threshold + 3, 0)
        
        self.vector_to_stream = blocks.vector_to_stream(
            gr.sizeof_float, fft_size)
        
        self.stream_to_vector2 = blocks.stream_to_vector(
            gr.sizeof_float, fft_size)
        
        # Connect spectrum path
        self.connect(self, self.stream_to_vector)
        self.connect(self.stream_to_vector, self.fft)
        self.connect(self.fft, self.complex_to_mag_squared)
        self.connect(self.complex_to_mag_squared, self.log10)
        self.connect(self.log10, self.avg)
        self.connect(self.avg, (self, 0))  # Spectrum output
        
        # Detection path
        self.connect(self.avg, self.vector_to_stream)
        self.connect(self.vector_to_stream, self.threshold_detector)
        self.connect(self.threshold_detector, self.stream_to_vector2)
        self.connect(self.stream_to_vector2, (self, 1))  # Detections output


class FilterBank(gr.hier_block2):
    """
    Multi-channel filter bank hierarchical block
    Splits signal into multiple frequency channels
    """
    
    def __init__(self, num_channels=4, sample_rate=1e6):
        """
        Initialize filter bank
        
        Args:
            num_channels: Number of channels
            sample_rate: Input sample rate
        """
        gr.hier_block2.__init__(
            self,
            "Filter Bank",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(num_channels, num_channels, gr.sizeof_gr_complex)
        )
        
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        
        # Calculate channel parameters
        channel_bw = sample_rate / num_channels
        transition_bw = channel_bw * 0.1
        
        # Create filters for each channel
        self.filters = []
        for i in range(num_channels):
            # Calculate center frequency for this channel
            center_freq = -sample_rate/2 + (i + 0.5) * channel_bw
            
            # Create bandpass filter
            if i == 0:
                # Lowpass for first channel
                taps = firdes.low_pass(
                    1.0, sample_rate, channel_bw/2, transition_bw)
            elif i == num_channels - 1:
                # Highpass for last channel
                taps = firdes.high_pass(
                    1.0, sample_rate, 
                    sample_rate/2 - channel_bw/2, transition_bw)
            else:
                # Bandpass for middle channels
                taps = firdes.band_pass(
                    1.0, sample_rate,
                    center_freq - channel_bw/2,
                    center_freq + channel_bw/2,
                    transition_bw)
            
            filt = gr_filter.fir_filter_ccc(1, taps)
            self.filters.append(filt)
            
            # Connect filter
            self.connect(self, filt)
            self.connect(filt, (self, i))


def create_test_flowgraph():
    """Create test flowgraph for hierarchical blocks"""
    
    class HierBlockTestFlowgraph(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self, "Hierarchical Block Test")
            
            # Parameters
            sample_rate = 100000
            audio_rate = 10000
            
            # Test signal source
            self.sig_source = analog.sig_source_c(
                sample_rate, analog.GR_COS_WAVE, 1000, 0.5, 0)
            
            # Hierarchical blocks
            self.channel = ChannelModel(
                noise_voltage=0.01,
                frequency_offset=0.001,
                time_offset=1.0001,
                taps=[1.0, 0.3, 0.1])
            
            self.agc = AGC(
                rate=1e-3,
                reference=1.0,
                gain=1.0,
                max_gain=100.0)
            
            # FM transceiver test
            self.audio_source = analog.sig_source_f(
                audio_rate, analog.GR_COS_WAVE, 440, 0.5, 0)
            
            self.fm_transceiver = FMTransceiver(
                audio_rate=audio_rate,
                if_rate=sample_rate,
                deviation=5000)
            
            # Spectrum sensor
            self.spectrum_sensor = SpectrumSensor(
                fft_size=256,
                sample_rate=sample_rate,
                center_freq=0,
                threshold=-40)
            
            # Filter bank
            self.filter_bank = FilterBank(
                num_channels=4,
                sample_rate=sample_rate)
            
            # Sinks
            self.null_sink_c = blocks.null_sink(gr.sizeof_gr_complex)
            self.null_sink_f = blocks.null_sink(gr.sizeof_float)
            self.null_sink_v1 = blocks.null_sink(gr.sizeof_float * 256)
            self.null_sink_v2 = blocks.null_sink(gr.sizeof_float * 256)
            
            # Connections
            # Channel and AGC test
            self.connect(self.sig_source, self.channel)
            self.connect(self.channel, self.agc)
            self.connect(self.agc, self.null_sink_c)
            
            # FM transceiver test
            self.connect(self.audio_source, self.fm_transceiver)
            self.connect((self.fm_transceiver, 0), self.null_sink_f)
            self.null_sink_f2 = blocks.null_sink(gr.sizeof_float)
            self.connect((self.fm_transceiver, 1), self.null_sink_f2)
            
            # Spectrum sensor test
            self.connect(self.sig_source, self.spectrum_sensor)
            self.connect((self.spectrum_sensor, 0), self.null_sink_v1)
            self.connect((self.spectrum_sensor, 1), self.null_sink_v2)
            
            # Filter bank test
            self.connect(self.sig_source, self.filter_bank)
            for i in range(4):
                sink = blocks.null_sink(gr.sizeof_gr_complex)
                self.connect((self.filter_bank, i), sink)
    
    return HierBlockTestFlowgraph()


def main():
    """Main function to demonstrate hierarchical blocks"""
    
    print("GNU Radio Hierarchical Block Examples")
    print("=" * 50)
    print("\nThis example demonstrates hierarchical blocks:")
    print("1. Channel Model - Complete channel simulation")
    print("2. AGC - Automatic gain control chain")
    print("3. FM Transceiver - Complete FM modem")
    print("4. Digital PLL - Phase-locked loop")
    print("5. Spectrum Sensor - FFT-based detection")
    print("6. Filter Bank - Multi-channel filtering")
    print("\nHierarchical blocks allow:")
    print("- Encapsulation of complex functionality")
    print("- Reusable components")
    print("- Clean flowgraph design")
    print("- Parameter abstraction")
    print("=" * 50)
    
    # Create and run test flowgraph
    tb = create_test_flowgraph()
    
    print("\nStarting hierarchical block test...")
    tb.start()
    
    # Test parameter updates
    time.sleep(2)
    print("\nIncreasing channel noise...")
    tb.channel.set_noise_voltage(0.1)
    
    time.sleep(2)
    print("Adding frequency offset...")
    tb.channel.set_frequency_offset(0.01)
    
    time.sleep(2)
    print("Adjusting AGC rate...")
    tb.agc.set_rate(1e-2)
    
    time.sleep(2)
    print("\nStopping...")
    tb.stop()
    tb.wait()
    
    print("\nHierarchical block test complete!")
    print("\nKey advantages demonstrated:")
    print("- Complex functionality in single blocks")
    print("- Dynamic parameter adjustment")
    print("- Clean signal flow abstraction")
    print("- Reusable across multiple projects")


if __name__ == '__main__':
    main()