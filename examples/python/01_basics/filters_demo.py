#!/usr/bin/env python3
"""
Digital Filters Demonstration
Shows various filter types and their effects on signals
Usage: python3 filters_demo.py --filter-type lowpass --cutoff 1000
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from gnuradio import gr
from gnuradio import blocks
from gnuradio import analog
from gnuradio import filter as gr_filter
from gnuradio.filter import firdes
from gnuradio import audio

class FilterDemo(gr.top_block):
    """Demonstrate various digital filter types"""
    
    def __init__(self, filter_type='lowpass', cutoff_freq=1000, 
                 sample_rate=48000, transition_width=100, 
                 filter_gain=1.0, window_type='hamming'):
        gr.top_block.__init__(self, "Filter Demonstration")
        
        self.sample_rate = sample_rate
        self.filter_type = filter_type
        self.cutoff_freq = cutoff_freq
        self.transition_width = transition_width
        
        # Create test signal: sum of multiple frequencies
        self.sig1 = analog.sig_source_f(sample_rate, analog.GR_COS_WAVE, 
                                        500, 0.3, 0, 0)  # 500 Hz
        self.sig2 = analog.sig_source_f(sample_rate, analog.GR_COS_WAVE, 
                                        1500, 0.3, 0, 0)  # 1500 Hz
        self.sig3 = analog.sig_source_f(sample_rate, analog.GR_COS_WAVE, 
                                        3000, 0.3, 0, 0)  # 3000 Hz
        self.noise = analog.noise_source_f(analog.GR_GAUSSIAN, 0.05, 0)
        
        # Add signals together
        self.adder = blocks.add_vff(1)
        self.connect(self.sig1, (self.adder, 0))
        self.connect(self.sig2, (self.adder, 1))
        self.connect(self.sig3, (self.adder, 2))
        self.connect(self.noise, (self.adder, 3))
        
        # Create filter based on type
        self.filter = self.create_filter(filter_type, cutoff_freq, 
                                        transition_width, filter_gain, 
                                        window_type)
        
        # Audio sink
        self.audio_sink = audio.sink(sample_rate, '', True)
        
        # File sinks for analysis
        self.original_file = blocks.file_sink(gr.sizeof_float, 
                                             '/tmp/original.dat', False)
        self.filtered_file = blocks.file_sink(gr.sizeof_float, 
                                             '/tmp/filtered.dat', False)
        
        # Connections
        self.connect(self.adder, self.filter)
        self.connect(self.filter, self.audio_sink)
        self.connect(self.adder, self.original_file)
        self.connect(self.filter, self.filtered_file)
        
        # Store filter taps for analysis
        self.filter_taps = self.filter.taps() if hasattr(self.filter, 'taps') else []
    
    def create_filter(self, filter_type, cutoff_freq, transition_width, 
                     gain, window_type):
        """Create filter based on specified type"""
        
        # Map window type string to constant
        window_map = {
            'hamming': gr_filter.window.WIN_HAMMING,
            'hann': gr_filter.window.WIN_HANN,
            'blackman': gr_filter.window.WIN_BLACKMAN,
            'rectangular': gr_filter.window.WIN_RECTANGULAR,
            'kaiser': gr_filter.window.WIN_KAISER,
            'blackman_harris': gr_filter.window.WIN_BLACKMAN_hARRIS
        }
        win = window_map.get(window_type.lower(), gr_filter.window.WIN_HAMMING)
        
        if filter_type == 'lowpass':
            taps = firdes.low_pass(
                gain,                    # gain
                self.sample_rate,        # sampling rate
                cutoff_freq,             # cutoff frequency
                transition_width,        # transition width
                win,                     # window type
                6.76                     # beta (for Kaiser window)
            )
            return gr_filter.fir_filter_fff(1, taps)
        
        elif filter_type == 'highpass':
            taps = firdes.high_pass(
                gain,
                self.sample_rate,
                cutoff_freq,
                transition_width,
                win,
                6.76
            )
            return gr_filter.fir_filter_fff(1, taps)
        
        elif filter_type == 'bandpass':
            # For bandpass, cutoff_freq is center, create band
            low_cutoff = cutoff_freq - 500
            high_cutoff = cutoff_freq + 500
            taps = firdes.band_pass(
                gain,
                self.sample_rate,
                low_cutoff,
                high_cutoff,
                transition_width,
                win,
                6.76
            )
            return gr_filter.fir_filter_fff(1, taps)
        
        elif filter_type == 'bandstop':
            # For bandstop, cutoff_freq is center, create notch
            low_cutoff = cutoff_freq - 200
            high_cutoff = cutoff_freq + 200
            taps = firdes.band_reject(
                gain,
                self.sample_rate,
                low_cutoff,
                high_cutoff,
                transition_width,
                win,
                6.76
            )
            return gr_filter.fir_filter_fff(1, taps)
        
        elif filter_type == 'allpass':
            # Simple delay (all-pass characteristic)
            taps = [0] * 10 + [1]  # 10-sample delay
            return gr_filter.fir_filter_fff(1, taps)
        
        elif filter_type == 'iir_lowpass':
            # IIR Butterworth lowpass filter
            # Note: GNU Radio's IIR filters need feed-forward and feedback taps
            from scipy import signal
            order = 4
            normalized_cutoff = cutoff_freq / (self.sample_rate / 2)
            b, a = signal.butter(order, normalized_cutoff, btype='low')
            return gr_filter.iir_filter_ffd(b.tolist(), a.tolist())
        
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
    
    def plot_frequency_response(self):
        """Plot filter frequency response"""
        if not self.filter_taps:
            print("No filter taps available for plotting")
            return
        
        # Calculate frequency response
        w, h = np.fft.fft(self.filter_taps, 2048), []
        freqs = np.linspace(0, self.sample_rate/2, len(w)//2)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Magnitude response
        magnitude_db = 20 * np.log10(np.abs(w[:len(w)//2]) + 1e-10)
        ax1.plot(freqs, magnitude_db)
        ax1.set_title(f'{self.filter_type.title()} Filter Frequency Response')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.grid(True)
        ax1.axhline(y=-3, color='r', linestyle='--', label='-3dB')
        ax1.axvline(x=self.cutoff_freq, color='g', linestyle='--', 
                   label=f'Cutoff: {self.cutoff_freq} Hz')
        ax1.legend()
        
        # Phase response
        phase = np.angle(w[:len(w)//2])
        ax2.plot(freqs, np.degrees(phase))
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (degrees)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_impulse_response(self):
        """Plot filter impulse response"""
        if not self.filter_taps:
            print("No filter taps available for plotting")
            return
        
        plt.figure(figsize=(10, 5))
        plt.stem(self.filter_taps[:100], basefmt=' ')
        plt.title(f'{self.filter_type.title()} Filter Impulse Response')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

class FilterAnalyzer:
    """Analyze and compare filtered signals"""
    
    @staticmethod
    def analyze_signals(original_file='/tmp/original.dat', 
                       filtered_file='/tmp/filtered.dat',
                       sample_rate=48000, duration=1.0):
        """Compare original and filtered signals"""
        
        # Read data
        samples_to_read = int(sample_rate * duration)
        
        try:
            original = np.fromfile(original_file, dtype=np.float32, 
                                  count=samples_to_read)
            filtered = np.fromfile(filtered_file, dtype=np.float32, 
                                  count=samples_to_read)
        except FileNotFoundError:
            print("Signal files not found. Run the filter first.")
            return
        
        if len(original) == 0 or len(filtered) == 0:
            print("No data available for analysis")
            return
        
        # Make sure both arrays have same length
        min_len = min(len(original), len(filtered))
        original = original[:min_len]
        filtered = filtered[:min_len]
        
        # Time domain plot
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        
        time = np.arange(min(1000, len(original))) / sample_rate
        
        # Time domain - first 1000 samples
        axes[0, 0].plot(time, original[:len(time)], 'b', alpha=0.7)
        axes[0, 0].set_title('Original Signal (Time Domain)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(time, filtered[:len(time)], 'r', alpha=0.7)
        axes[0, 1].set_title('Filtered Signal (Time Domain)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True)
        
        # Frequency domain
        fft_size = min(8192, len(original))
        freqs = np.fft.fftfreq(fft_size, 1/sample_rate)[:fft_size//2]
        
        original_fft = np.fft.fft(original[:fft_size])
        filtered_fft = np.fft.fft(filtered[:fft_size])
        
        axes[1, 0].plot(freqs, 20*np.log10(np.abs(original_fft[:fft_size//2]) + 1e-10))
        axes[1, 0].set_title('Original Signal (Frequency Domain)')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Magnitude (dB)')
        axes[1, 0].grid(True)
        axes[1, 0].set_xlim([0, sample_rate/2])
        
        axes[1, 1].plot(freqs, 20*np.log10(np.abs(filtered_fft[:fft_size//2]) + 1e-10))
        axes[1, 1].set_title('Filtered Signal (Frequency Domain)')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Magnitude (dB)')
        axes[1, 1].grid(True)
        axes[1, 1].set_xlim([0, sample_rate/2])
        
        # Spectrogram
        from scipy import signal
        
        f, t, Sxx_orig = signal.spectrogram(original, sample_rate, nperseg=256)
        axes[2, 0].pcolormesh(t, f, 10*np.log10(Sxx_orig + 1e-10))
        axes[2, 0].set_title('Original Signal Spectrogram')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Frequency (Hz)')
        
        f, t, Sxx_filt = signal.spectrogram(filtered, sample_rate, nperseg=256)
        axes[2, 1].pcolormesh(t, f, 10*np.log10(Sxx_filt + 1e-10))
        axes[2, 1].set_title('Filtered Signal Spectrogram')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Frequency (Hz)')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("\nSignal Statistics:")
        print("-" * 40)
        print(f"Original - RMS: {np.sqrt(np.mean(original**2)):.4f}")
        print(f"Filtered - RMS: {np.sqrt(np.mean(filtered**2)):.4f}")
        print(f"Original - Peak: {np.max(np.abs(original)):.4f}")
        print(f"Filtered - Peak: {np.max(np.abs(filtered)):.4f}")
        
        # Calculate attenuation at specific frequencies
        test_freqs = [500, 1500, 3000]
        print("\nFrequency Attenuation:")
        for freq in test_freqs:
            idx = int(freq * fft_size / sample_rate)
            if idx < len(original_fft)//2:
                orig_mag = np.abs(original_fft[idx])
                filt_mag = np.abs(filtered_fft[idx])
                if orig_mag > 0:
                    atten = 20 * np.log10(filt_mag / orig_mag)
                    print(f"  {freq} Hz: {atten:.1f} dB")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='GNU Radio Digital Filters Demonstration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--filter-type', 
                       choices=['lowpass', 'highpass', 'bandpass', 
                               'bandstop', 'allpass', 'iir_lowpass'],
                       default='lowpass',
                       help='Type of filter to apply')
    
    parser.add_argument('--cutoff', 
                       type=float,
                       default=1000,
                       help='Cutoff frequency in Hz')
    
    parser.add_argument('--transition', 
                       type=float,
                       default=100,
                       help='Transition width in Hz')
    
    parser.add_argument('--gain', 
                       type=float,
                       default=1.0,
                       help='Filter gain')
    
    parser.add_argument('--window', 
                       choices=['hamming', 'hann', 'blackman', 'rectangular', 
                               'kaiser', 'blackman_harris'],
                       default='hamming',
                       help='Window type for FIR filter')
    
    parser.add_argument('--sample-rate', 
                       type=int,
                       default=48000,
                       help='Sample rate in Hz')
    
    parser.add_argument('--duration', 
                       type=float,
                       default=5,
                       help='Duration in seconds')
    
    parser.add_argument('--analyze', 
                       action='store_true',
                       help='Analyze and plot results')
    
    parser.add_argument('--plot-response', 
                       action='store_true',
                       help='Plot filter frequency response')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Create filter demonstration
    tb = FilterDemo(
        filter_type=args.filter_type,
        cutoff_freq=args.cutoff,
        sample_rate=args.sample_rate,
        transition_width=args.transition,
        filter_gain=args.gain,
        window_type=args.window
    )
    
    # Print info
    print("GNU Radio Filter Demonstration")
    print("-" * 40)
    print(f"Filter Type:       {args.filter_type}")
    print(f"Cutoff Frequency:  {args.cutoff} Hz")
    print(f"Transition Width:  {args.transition} Hz")
    print(f"Window Type:       {args.window}")
    print(f"Sample Rate:       {args.sample_rate} Hz")
    print(f"Filter Taps:       {len(tb.filter_taps) if tb.filter_taps else 'N/A'}")
    print("-" * 40)
    print("Test signal contains: 500 Hz, 1500 Hz, 3000 Hz + noise")
    print(f"Running for {args.duration} seconds...")
    print("Press Ctrl+C to stop")
    
    # Start the flowgraph
    tb.start()
    
    # Plot filter response if requested
    if args.plot_response:
        tb.plot_frequency_response()
        tb.plot_impulse_response()
    
    try:
        import time
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nStopping...")
    
    # Stop the flowgraph
    tb.stop()
    tb.wait()
    
    # Analyze if requested
    if args.analyze:
        print("\nAnalyzing filtered signal...")
        analyzer = FilterAnalyzer()
        analyzer.analyze_signals(sample_rate=args.sample_rate, 
                               duration=min(1.0, args.duration))

if __name__ == '__main__':
    main()
