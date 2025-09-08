#!/usr/bin/env python3
"""
FFT Analysis and Spectrum Visualization
Demonstrates FFT operations and spectrum analysis in GNU Radio
Usage: python3 fft_analysis.py --fft-size 1024 --window hamming
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gnuradio import gr
from gnuradio import blocks
from gnuradio import analog
from gnuradio import fft
from gnuradio.filter import firdes
from gnuradio.fft import window
import time
import threading

class SpectrumAnalyzer(gr.top_block):
    """Real-time spectrum analyzer using FFT"""
    
    def __init__(self, sample_rate=1e6, fft_size=1024, 
                 window_type='hamming', averaging=0.8):
        gr.top_block.__init__(self, "Spectrum Analyzer")
        
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.window_type = window_type
        self.averaging = averaging
        
        # Create window
        self.window = self.create_window(window_type, fft_size)
        
        # Create test signal with multiple components
        self.create_test_signal()
        
        # Stream to vector
        self.s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, fft_size)
        
        # FFT block
        self.fft_block = fft.fft_vcc(fft_size, True, self.window, True, 1)
        
        # Complex to magnitude squared
        self.c2mag2 = blocks.complex_to_mag_squared(fft_size)
        
        # Averaging filter (single pole IIR)
        if averaging > 0:
            self.avg = blocks.single_pole_iir_filter_ff(averaging, fft_size)
        else:
            self.avg = blocks.multiply_const_vff([1.0] * fft_size)
        
        # Convert to dB
        self.log10 = blocks.nlog10_ff(10, fft_size, -20)  # 10*log10() with offset
        
        # Probe to get FFT data
        self.probe = blocks.probe_signal_vf(fft_size)
        
        # File sink for recording
        self.file_sink = blocks.file_sink(gr.sizeof_float * fft_size, 
                                         '/tmp/spectrum.dat', False)
        self.file_sink.set_unbuffered(False)
        
        # Connections
        self.connect(self.test_signal, self.s2v)
        self.connect(self.s2v, self.fft_block)
        self.connect(self.fft_block, self.c2mag2)
        self.connect(self.c2mag2, self.avg)
        self.connect(self.avg, self.log10)
        self.connect(self.log10, self.probe)
        self.connect(self.log10, self.file_sink)
        
        # Variables for spectrum data
        self.spectrum_data = np.zeros(fft_size)
        self.freq_bins = np.fft.fftshift(
            np.fft.fftfreq(fft_size, 1/sample_rate))
    
    def create_test_signal(self):
        """Create a complex test signal with multiple components"""
        # Component 1: Single tone
        self.tone1 = analog.sig_source_c(self.sample_rate, 
                                         analog.GR_COS_WAVE, 
                                         100e3, 0.5, 0, 0)
        
        # Component 2: Another tone
        self.tone2 = analog.sig_source_c(self.sample_rate, 
                                         analog.GR_COS_WAVE, 
                                         -200e3, 0.3, 0, 0)
        
        # Component 3: Chirp signal
        self.chirp = analog.sig_source_c(self.sample_rate, 
                                         analog.GR_COS_WAVE, 
                                         50e3, 0.2, 0, 0)
        
        # Noise
        self.noise = blocks.noise_source_c(blocks.GR_GAUSSIAN, 0.01, 0)
        
        # Add all components
        self.adder = blocks.add_vcc(1)
        self.connect(self.tone1, (self.adder, 0))
        self.connect(self.tone2, (self.adder, 1))
        self.connect(self.chirp, (self.adder, 2))
        self.connect(self.noise, (self.adder, 3))
        
        # Throttle for controlled rate
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, self.sample_rate)
        self.connect(self.adder, self.throttle)
        
        self.test_signal = self.throttle
    
    def create_window(self, window_type, size):
        """Create window function for FFT"""
        window_map = {
            'hamming': window.hamming,
            'hann': window.hann,
            'blackman': window.blackman,
            'blackman_harris': window.blackman_harris,
            'rectangular': window.rectangular,
            'kaiser': lambda n: window.kaiser(n, 6.76),
            'bartlett': window.bartlett
        }
        
        if window_type.lower() in window_map:
            return window_map[window_type.lower()](size)
        else:
            print(f"Unknown window type: {window_type}, using Hamming")
            return window.hamming(size)
    
    def get_spectrum(self):
        """Get current spectrum data"""
        data = self.probe.level()
        if len(data) == self.fft_size:
            # FFT shift to center DC
            self.spectrum_data = np.fft.fftshift(data)
        return self.spectrum_data
    
    def update_chirp_frequency(self):
        """Update chirp frequency for sweeping"""
        thread_running = True
        sweep_range = 200e3
        sweep_period = 2.0
        
        while thread_running:
            try:
                for t in np.linspace(0, sweep_period, 100):
                    freq = -sweep_range/2 + (sweep_range * t / sweep_period)
                    self.chirp.set_frequency(freq)
                    time.sleep(sweep_period / 100)
            except:
                thread_running = False
                break

class FFTProcessor:
    """FFT processing and analysis utilities"""
    
    @staticmethod
    def compute_psd(signal, sample_rate, fft_size=None, 
                    window_type='hamming', overlap=0.5):
        """Compute Power Spectral Density using Welch's method"""
        if fft_size is None:
            fft_size = min(len(signal), 1024)
        
        from scipy import signal as scipy_signal
        
        # Compute PSD
        freqs, psd = scipy_signal.welch(signal, sample_rate, 
                                        nperseg=fft_size,
                                        window=window_type,
                                        noverlap=int(fft_size * overlap))
        
        return freqs, 10 * np.log10(psd + 1e-10)
    
    @staticmethod
    def compute_spectrogram(signal, sample_rate, fft_size=256, 
                           window_type='hamming', overlap=0.5):
        """Compute spectrogram of signal"""
        from scipy import signal as scipy_signal
        
        # Compute spectrogram
        f, t, Sxx = scipy_signal.spectrogram(signal, sample_rate,
                                            window=window_type,
                                            nperseg=fft_size,
                                            noverlap=int(fft_size * overlap))
        
        return f, t, 10 * np.log10(Sxx + 1e-10)
    
    @staticmethod
    def find_peaks(spectrum, freqs, threshold=-40):
        """Find peaks in spectrum"""
        from scipy.signal import find_peaks
        
        # Find peaks above threshold
        peaks, properties = find_peaks(spectrum, height=threshold, 
                                      distance=10, prominence=10)
        
        peak_freqs = freqs[peaks]
        peak_powers = spectrum[peaks]
        
        return peak_freqs, peak_powers, peaks
    
    @staticmethod
    def calculate_snr(signal, signal_freq, noise_bw, sample_rate):
        """Calculate SNR from spectrum"""
        # Compute PSD
        freqs, psd = FFTProcessor.compute_psd(signal, sample_rate)
        
        # Find signal peak
        signal_idx = np.argmin(np.abs(freqs - signal_freq))
        signal_power = psd[signal_idx]
        
        # Calculate noise power (excluding signal)
        noise_mask = np.abs(freqs - signal_freq) > noise_bw
        noise_power = np.mean(psd[noise_mask])
        
        snr = signal_power - noise_power
        return snr
    
    @staticmethod
    def compute_phase_spectrum(signal, fft_size=None):
        """Compute phase spectrum"""
        if fft_size is None:
            fft_size = len(signal)
        
        # Zero-pad if necessary
        if len(signal) < fft_size:
            signal = np.pad(signal, (0, fft_size - len(signal)))
        
        # Compute FFT
        spectrum = np.fft.fft(signal[:fft_size])
        
        # Get phase
        phase = np.angle(spectrum)
        
        # Unwrap phase
        phase_unwrapped = np.unwrap(phase)
        
        return phase, phase_unwrapped

class RealtimeSpectrumPlotter:
    """Real-time spectrum plotting with matplotlib"""
    
    def __init__(self, spectrum_analyzer):
        self.analyzer = spectrum_analyzer
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Setup spectrum plot
        self.ax1.set_title('Real-time FFT Spectrum')
        self.ax1.set_xlabel('Frequency (kHz)')
        self.ax1.set_ylabel('Power (dB)')
        self.ax1.grid(True)
        self.ax1.set_ylim([-80, 0])
        
        freq_khz = self.analyzer.freq_bins / 1000
        self.line1, = self.ax1.plot(freq_khz, np.zeros_like(freq_khz))
        
        # Setup waterfall plot
        self.ax2.set_title('Waterfall Display')
        self.ax2.set_xlabel('Frequency (kHz)')
        self.ax2.set_ylabel('Time (seconds)')
        
        self.waterfall_data = np.zeros((50, len(freq_khz)))
        self.waterfall_img = self.ax2.imshow(self.waterfall_data, 
                                            aspect='auto',
                                            extent=[freq_khz[0], freq_khz[-1], 
                                                   0, 5],
                                            cmap='viridis',
                                            vmin=-80, vmax=0)
        
        self.time_counter = 0
        
    def update_plot(self, frame):
        """Update plot with new spectrum data"""
        spectrum = self.analyzer.get_spectrum()
        
        # Update spectrum plot
        self.line1.set_ydata(spectrum)
        
        # Update waterfall
        self.waterfall_data = np.roll(self.waterfall_data, 1, axis=0)
        self.waterfall_data[0, :] = spectrum
        self.waterfall_img.set_data(self.waterfall_data)
        
        self.time_counter += 1
        
        # Find and annotate peaks
        if self.time_counter % 10 == 0:  # Every 10 frames
            self.ax1.clear()
            self.ax1.set_title('Real-time FFT Spectrum')
            self.ax1.set_xlabel('Frequency (kHz)')
            self.ax1.set_ylabel('Power (dB)')
            self.ax1.grid(True)
            self.ax1.set_ylim([-80, 0])
            
            freq_khz = self.analyzer.freq_bins / 1000
            self.ax1.plot(freq_khz, spectrum)
            
            # Find peaks
            peak_freqs, peak_powers, peak_indices = FFTProcessor.find_peaks(
                spectrum, self.analyzer.freq_bins, threshold=-40)
            
            # Annotate peaks
            for freq, power in zip(peak_freqs, peak_powers):
                self.ax1.annotate(f'{freq/1000:.1f} kHz',
                                xy=(freq/1000, power),
                                xytext=(freq/1000, power + 5),
                                fontsize=8,
                                ha='center')
        
        return self.line1, self.waterfall_img
    
    def start(self):
        """Start real-time plotting"""
        self.ani = FuncAnimation(self.fig, self.update_plot, 
                               interval=100, blit=False)
        plt.show()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='GNU Radio FFT Analysis and Spectrum Visualization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--fft-size', 
                       type=int,
                       default=1024,
                       help='FFT size (power of 2)')
    
    parser.add_argument('--window', 
                       choices=['hamming', 'hann', 'blackman', 'blackman_harris',
                               'rectangular', 'kaiser', 'bartlett'],
                       default='hamming',
                       help='Window type for FFT')
    
    parser.add_argument('--sample-rate', 
                       type=float,
                       default=1e6,
                       help='Sample rate in Hz')
    
    parser.add_argument('--averaging', 
                       type=float,
                       default=0.8,
                       help='Spectrum averaging factor (0-1)')
    
    parser.add_argument('--duration', 
                       type=float,
                       default=10,
                       help='Analysis duration in seconds')
    
    parser.add_argument('--realtime', 
                       action='store_true',
                       help='Show real-time spectrum plot')
    
    parser.add_argument('--analyze-file', 
                       type=str,
                       help='Analyze IQ file instead of test signal')
    
    return parser.parse_args()

def analyze_file(filename, sample_rate, fft_size, window_type):
    """Analyze IQ file with FFT"""
    print(f"Analyzing file: {filename}")
    
    # Read IQ data
    try:
        data = np.fromfile(filename, dtype=np.complex64)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return
    
    if len(data) == 0:
        print("File is empty")
        return
    
    print(f"Loaded {len(data):,} samples")
    
    # Compute PSD
    freqs, psd = FFTProcessor.compute_psd(data, sample_rate, 
                                         fft_size, window_type)
    
    # Find peaks
    peak_freqs, peak_powers, _ = FFTProcessor.find_peaks(psd, freqs)
    
    # Compute spectrogram
    f, t, Sxx = FFTProcessor.compute_spectrogram(data[:int(sample_rate)], 
                                                sample_rate, 
                                                fft_size//4, window_type)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # PSD plot
    axes[0, 0].plot(freqs/1e3, psd)
    axes[0, 0].set_title('Power Spectral Density')
    axes[0, 0].set_xlabel('Frequency (kHz)')
    axes[0, 0].set_ylabel('Power (dB)')
    axes[0, 0].grid(True)
    
    # Mark peaks
    for freq, power in zip(peak_freqs, peak_powers):
        axes[0, 0].plot(freq/1e3, power, 'ro')
        axes[0, 0].annotate(f'{freq/1e3:.1f} kHz',
                          xy=(freq/1e3, power),
                          xytext=(freq/1e3, power + 5),
                          fontsize=8)
    
    # Time domain (first 1000 samples)
    samples_to_plot = min(1000, len(data))
    time_axis = np.arange(samples_to_plot) / sample_rate * 1e6  # microseconds
    axes[0, 1].plot(time_axis, np.real(data[:samples_to_plot]), 'b', 
                   label='I', alpha=0.7)
    axes[0, 1].plot(time_axis, np.imag(data[:samples_to_plot]), 'r', 
                   label='Q', alpha=0.7)
    axes[0, 1].set_title('Time Domain (I/Q)')
    axes[0, 1].set_xlabel('Time (μs)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Spectrogram
    im = axes[1, 0].pcolormesh(t, f/1e3, Sxx, shading='gouraud',
                              cmap='viridis')
    axes[1, 0].set_title('Spectrogram')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Frequency (kHz)')
    plt.colorbar(im, ax=axes[1, 0], label='Power (dB)')
    
    # Phase spectrum
    phase, phase_unwrapped = FFTProcessor.compute_phase_spectrum(
        data[:fft_size])
    
    phase_freqs = np.fft.fftfreq(fft_size, 1/sample_rate)
    axes[1, 1].plot(phase_freqs[:fft_size//2]/1e3, 
                   phase[:fft_size//2], 'b', label='Wrapped', alpha=0.7)
    axes[1, 1].plot(phase_freqs[:fft_size//2]/1e3, 
                   phase_unwrapped[:fft_size//2], 'r', 
                   label='Unwrapped', alpha=0.7)
    axes[1, 1].set_title('Phase Spectrum')
    axes[1, 1].set_xlabel('Frequency (kHz)')
    axes[1, 1].set_ylabel('Phase (radians)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nSpectrum Analysis Results:")
    print("-" * 40)
    print(f"Peak Frequencies: {[f'{f/1e3:.1f} kHz' for f in peak_freqs]}")
    print(f"Peak Powers: {[f'{p:.1f} dB' for p in peak_powers]}")
    print(f"Noise Floor: {np.mean(psd):.1f} dB")
    print(f"Dynamic Range: {np.max(psd) - np.min(psd):.1f} dB")

def main():
    args = parse_arguments()
    
    # Validate FFT size is power of 2
    if not (args.fft_size & (args.fft_size - 1)) == 0:
        print("Warning: FFT size should be a power of 2 for best performance")
    
    if args.analyze_file:
        # Analyze existing file
        analyze_file(args.analyze_file, args.sample_rate, 
                    args.fft_size, args.window)
    else:
        # Create spectrum analyzer
        analyzer = SpectrumAnalyzer(
            sample_rate=args.sample_rate,
            fft_size=args.fft_size,
            window_type=args.window,
            averaging=args.averaging
        )
        
        print("GNU Radio FFT Spectrum Analyzer")
        print("-" * 40)
        print(f"Sample Rate: {args.sample_rate/1e6:.1f} MHz")
        print(f"FFT Size: {args.fft_size}")
        print(f"Window: {args.window}")
        print(f"Averaging: {args.averaging}")
        print(f"Frequency Resolution: {args.sample_rate/args.fft_size:.1f} Hz")
        print("-" * 40)
        
        # Start analyzer
        analyzer.start()
        
        # Start chirp update thread
        chirp_thread = threading.Thread(target=analyzer.update_chirp_frequency)
        chirp_thread.daemon = True
        chirp_thread.start()
        
        if args.realtime:
            # Real-time plotting
            print("Starting real-time spectrum display...")
            plotter = RealtimeSpectrumPlotter(analyzer)
            plotter.start()
        else:
            # Just run for specified duration
            print(f"Running for {args.duration} seconds...")
            print("Spectrum data being saved to /tmp/spectrum.dat")
            
            try:
                time.sleep(args.duration)
            except KeyboardInterrupt:
                print("\nStopping...")
        
        # Stop analyzer
        analyzer.stop()
        analyzer.wait()
        
        print("\nAnalysis complete!")

if __name__ == '__main__':
    main()