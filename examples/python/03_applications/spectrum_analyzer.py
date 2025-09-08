#!/usr/bin/env python3
"""
Real-time Spectrum Analyzer Application
Wideband spectrum analyzer with waterfall display
Usage: python3 spectrum_analyzer.py --start-freq 88 --stop-freq 108 --source rtlsdr
"""

import argparse
import sys
import numpy as np
import time
import threading
from gnuradio import gr
from gnuradio import blocks
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import filter as gr_filter
from gnuradio.filter import firdes
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

class SpectrumAnalyzer(gr.top_block):
    """
    Wideband spectrum analyzer with frequency hopping
    """
    
    def __init__(self, start_freq=88e6, stop_freq=108e6, 
                 sample_rate=2e6, fft_size=2048,
                 source_type='simulation', device_args='',
                 gain=30, averaging=0.8):
        """
        Initialize spectrum analyzer
        
        Args:
            start_freq: Start frequency in Hz
            stop_freq: Stop frequency in Hz
            sample_rate: Sample rate in Hz
            fft_size: FFT size
            source_type: 'rtlsdr', 'hackrf', 'simulation'
            device_args: Device-specific arguments
            gain: RF gain
            averaging: Spectrum averaging factor (0-1)
        """
        gr.top_block.__init__(self, "Spectrum Analyzer")
        
        # Parameters
        self.start_freq = start_freq
        self.stop_freq = stop_freq
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.source_type = source_type
        self.gain = gain
        self.averaging = averaging
        
        # Calculate sweep parameters
        self.usable_bw = sample_rate * 0.8  # 80% usable bandwidth
        self.num_steps = int(np.ceil((stop_freq - start_freq) / self.usable_bw))
        self.current_freq = start_freq + self.usable_bw / 2
        self.freq_step = self.usable_bw
        
        # Create source
        if source_type == 'simulation':
            self.source = self.create_simulation_source()
        elif source_type == 'rtlsdr':
            self.source = self.create_rtlsdr_source(device_args)
        elif source_type == 'hackrf':
            self.source = self.create_hackrf_source(device_args)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        # FFT processing chain
        self.setup_fft_chain()
        
        # Peak detection
        self.setup_peak_detector()
        
        # Data storage
        self.spectrum_data = np.zeros(fft_size)
        self.waterfall_data = deque(maxlen=50)
        self.peak_frequencies = []
        self.peak_powers = []
        
        # Sweep control
        self.sweep_thread = None
        self.sweep_running = False
        self.sweep_data = {}
        
        # Connect flowgraph
        self.connect_flowgraph()
    
    def create_simulation_source(self):
        """Create simulation source with test signals"""
        from gnuradio import analog
        
        # Multiple test signals
        sig1 = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE, 
                                   100e3, 0.5, 0)
        sig2 = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE, 
                                   -200e3, 0.3, 0)
        sig3 = analog.sig_source_c(self.sample_rate, analog.GR_COS_WAVE, 
                                   500e3, 0.2, 0)
        noise = analog.noise_source_c(analog.GR_GAUSSIAN, 0.01, 0)
        
        adder = blocks.add_cc()
        throttle = blocks.throttle(gr.sizeof_gr_complex, self.sample_rate)
        
        # Create hierarchical block
        sim_source = gr.hier_block2(
            "Simulation Source",
            gr.io_signature(0, 0, 0),
            gr.io_signature(1, 1, gr.sizeof_gr_complex)
        )
        
        sim_source.connect(sig1, (adder, 0))
        sim_source.connect(sig2, (adder, 1))
        sim_source.connect(sig3, (adder, 2))
        sim_source.connect(noise, (adder, 3))
        sim_source.connect(adder, throttle)
        sim_source.connect(throttle, sim_source)
        
        return sim_source
    
    def create_rtlsdr_source(self, device_args):
        """Create RTL-SDR source"""
        try:
            from gnuradio import osmosdr
        except ImportError:
            print("Warning: gr-osmosdr not installed, using simulation")
            return self.create_simulation_source()
        
        rtlsdr = osmosdr.source(device_args or "rtl=0")
        rtlsdr.set_sample_rate(self.sample_rate)
        rtlsdr.set_center_freq(self.current_freq, 0)
        rtlsdr.set_freq_corr(0, 0)
        rtlsdr.set_gain(self.gain, 0)
        rtlsdr.set_if_gain(20, 0)
        rtlsdr.set_bb_gain(20, 0)
        
        return rtlsdr
    
    def create_hackrf_source(self, device_args):
        """Create HackRF source"""
        try:
            from gnuradio import osmosdr
        except ImportError:
            print("Warning: gr-osmosdr not installed, using simulation")
            return self.create_simulation_source()
        
        hackrf = osmosdr.source(device_args or "hackrf=0")
        hackrf.set_sample_rate(self.sample_rate)
        hackrf.set_center_freq(self.current_freq, 0)
        hackrf.set_freq_corr(0, 0)
        hackrf.set_gain(14, 0)
        hackrf.set_if_gain(self.gain, 0)
        hackrf.set_bb_gain(20, 0)
        
        return hackrf
    
    def setup_fft_chain(self):
        """Setup FFT processing chain"""
        # Stream to vector
        self.s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, self.fft_size)
        
        # Window
        fft_window = window.blackmanharris(self.fft_size)
        
        # FFT
        self.fft_block = fft.fft_vcc(self.fft_size, True, fft_window, True, 1)
        
        # Complex to magnitude squared
        self.c2mag2 = blocks.complex_to_mag_squared(self.fft_size)
        
        # Single-pole IIR filter for averaging
        self.avg_filter = blocks.single_pole_iir_filter_ff(
            self.averaging, self.fft_size)
        
        # Convert to dB
        self.log10 = blocks.nlog10_ff(10, self.fft_size, -20)
        
        # Probe for getting data
        self.probe = blocks.probe_signal_vf(self.fft_size)
    
    def setup_peak_detector(self):
        """Setup peak detection"""
        # Threshold for peak detection
        self.threshold = -60  # dBm
        
        # Peak hold
        self.peak_hold = blocks.max_ff(self.fft_size, self.fft_size)
        
    def connect_flowgraph(self):
        """Connect all blocks"""
        self.connect(self.source, self.s2v)
        self.connect(self.s2v, self.fft_block)
        self.connect(self.fft_block, self.c2mag2)
        self.connect(self.c2mag2, self.avg_filter)
        self.connect(self.avg_filter, self.log10)
        self.connect(self.log10, self.probe)
    
    def get_spectrum(self):
        """Get current spectrum data"""
        data = self.probe.level()
        if len(data) == self.fft_size:
            self.spectrum_data = np.fft.fftshift(data)
        return self.spectrum_data
    
    def find_peaks(self, threshold=None):
        """Find peaks in current spectrum"""
        if threshold is None:
            threshold = self.threshold
        
        spectrum = self.get_spectrum()
        
        # Simple peak detection
        peaks = []
        for i in range(1, len(spectrum) - 1):
            if (spectrum[i] > threshold and 
                spectrum[i] > spectrum[i-1] and 
                spectrum[i] > spectrum[i+1]):
                
                # Convert bin to frequency
                freq_offset = (i - self.fft_size/2) * self.sample_rate / self.fft_size
                freq = self.current_freq + freq_offset
                peaks.append((freq, spectrum[i]))
        
        return peaks
    
    def set_center_freq(self, freq):
        """Set center frequency"""
        self.current_freq = freq
        if self.source_type != 'simulation':
            self.source.set_center_freq(freq, 0)
    
    def set_gain(self, gain):
        """Set RF gain"""
        self.gain = gain
        if self.source_type != 'simulation':
            self.source.set_gain(gain, 0)
    
    def start_sweep(self):
        """Start frequency sweep"""
        self.sweep_running = True
        self.sweep_thread = threading.Thread(target=self.sweep_worker)
        self.sweep_thread.daemon = True
        self.sweep_thread.start()
    
    def stop_sweep(self):
        """Stop frequency sweep"""
        self.sweep_running = False
        if self.sweep_thread:
            self.sweep_thread.join()
    
    def sweep_worker(self):
        """Worker thread for frequency sweeping"""
        print(f"Starting sweep from {self.start_freq/1e6:.1f} to "
              f"{self.stop_freq/1e6:.1f} MHz")
        
        while self.sweep_running:
            for step in range(self.num_steps):
                if not self.sweep_running:
                    break
                
                # Calculate frequency for this step
                freq = self.start_freq + step * self.freq_step + self.freq_step/2
                
                # Tune to frequency
                self.set_center_freq(freq)
                
                # Wait for settling
                time.sleep(0.1)
                
                # Get spectrum
                spectrum = self.get_spectrum()
                
                # Store data
                self.sweep_data[freq] = spectrum
                
                # Find peaks
                peaks = self.find_peaks()
                for peak_freq, peak_power in peaks:
                    self.peak_frequencies.append(peak_freq)
                    self.peak_powers.append(peak_power)
                
                # Update waterfall
                self.waterfall_data.append(spectrum)
                
                print(f"  Scanning {freq/1e6:.1f} MHz, "
                      f"found {len(peaks)} peaks")
        
        print("Sweep complete")
    
    def get_full_spectrum(self):
        """Combine sweep data into full spectrum"""
        if not self.sweep_data:
            return None, None
        
        # Combine all frequency bins
        all_freqs = []
        all_powers = []
        
        for center_freq in sorted(self.sweep_data.keys()):
            spectrum = self.sweep_data[center_freq]
            freq_bins = np.linspace(
                center_freq - self.sample_rate/2,
                center_freq + self.sample_rate/2,
                self.fft_size)
            
            all_freqs.extend(freq_bins)
            all_powers.extend(spectrum)
        
        return np.array(all_freqs), np.array(all_powers)


class SpectrumPlotter:
    """Real-time spectrum plotting"""
    
    def __init__(self, analyzer):
        """Initialize plotter
        
        Args:
            analyzer: SpectrumAnalyzer instance
        """
        self.analyzer = analyzer
        
        # Setup plot
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            3, 1, figsize=(12, 10))
        
        # Spectrum plot
        self.ax1.set_title('Real-time Spectrum')
        self.ax1.set_xlabel('Frequency (MHz)')
        self.ax1.set_ylabel('Power (dB)')
        self.ax1.grid(True)
        self.ax1.set_ylim([-100, 0])
        
        freq_mhz = np.linspace(
            -analyzer.sample_rate/2e6,
            analyzer.sample_rate/2e6,
            analyzer.fft_size)
        self.line1, = self.ax1.plot(freq_mhz, np.zeros(analyzer.fft_size))
        
        # Waterfall plot
        self.ax2.set_title('Waterfall Display')
        self.ax2.set_xlabel('Frequency (MHz)')
        self.ax2.set_ylabel('Time (seconds)')
        
        self.waterfall_img = self.ax2.imshow(
            np.zeros((50, analyzer.fft_size)),
            aspect='auto',
            extent=[freq_mhz[0], freq_mhz[-1], 0, 5],
            cmap='viridis',
            vmin=-100, vmax=0)
        
        # Peak histogram
        self.ax3.set_title('Detected Signals')
        self.ax3.set_xlabel('Frequency (MHz)')
        self.ax3.set_ylabel('Occurrences')
        self.ax3.grid(True)
        
        self.peak_bars = None
        
        # Animation
        self.frame_count = 0
    
    def update_plot(self, frame):
        """Update plot with new data"""
        # Get current spectrum
        spectrum = self.analyzer.get_spectrum()
        
        # Update spectrum plot
        center_freq_mhz = self.analyzer.current_freq / 1e6
        freq_mhz = np.linspace(
            center_freq_mhz - self.analyzer.sample_rate/2e6,
            center_freq_mhz + self.analyzer.sample_rate/2e6,
            self.analyzer.fft_size)
        
        self.line1.set_xdata(freq_mhz)
        self.line1.set_ydata(spectrum)
        self.ax1.set_xlim([freq_mhz[0], freq_mhz[-1]])
        self.ax1.set_title(f'Spectrum at {center_freq_mhz:.1f} MHz')
        
        # Update waterfall
        if len(self.analyzer.waterfall_data) > 0:
            waterfall_array = np.array(list(self.analyzer.waterfall_data))
            self.waterfall_img.set_data(waterfall_array)
            self.waterfall_img.set_extent(
                [freq_mhz[0], freq_mhz[-1], 0, len(waterfall_array) * 0.1])
        
        # Update peak histogram
        if len(self.analyzer.peak_frequencies) > 0:
            # Bin the peaks
            hist, bins = np.histogram(
                np.array(self.analyzer.peak_frequencies) / 1e6,
                bins=50,
                range=(self.analyzer.start_freq/1e6, 
                       self.analyzer.stop_freq/1e6))
            
            if self.peak_bars is None:
                self.peak_bars = self.ax3.bar(
                    bins[:-1], hist, width=np.diff(bins)[0])
            else:
                for bar, h in zip(self.peak_bars, hist):
                    bar.set_height(h)
        
        self.frame_count += 1
        
        # Find and annotate current peaks
        if self.frame_count % 10 == 0:
            peaks = self.analyzer.find_peaks()
            
            # Clear old annotations
            for txt in self.ax1.texts:
                txt.remove()
            
            # Add new annotations
            for freq, power in peaks[:5]:  # Top 5 peaks
                freq_mhz = freq / 1e6
                if freq_mhz >= freq_mhz[0] and freq_mhz <= freq_mhz[-1]:
                    self.ax1.annotate(
                        f'{freq_mhz:.2f}',
                        xy=(freq_mhz, power),
                        xytext=(freq_mhz, power + 5),
                        fontsize=8,
                        ha='center')
        
        return self.line1, self.waterfall_img
    
    def start(self):
        """Start animated plotting"""
        self.ani = FuncAnimation(
            self.fig, self.update_plot,
            interval=100, blit=False)
        plt.show()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='GNU Radio Spectrum Analyzer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--start-freq', 
                       type=float,
                       default=88,
                       help='Start frequency in MHz')
    
    parser.add_argument('--stop-freq', 
                       type=float,
                       default=108,
                       help='Stop frequency in MHz')
    
    parser.add_argument('-s', '--sample-rate', 
                       type=float,
                       default=2e6,
                       help='Sample rate in Hz')
    
    parser.add_argument('--fft-size', 
                       type=int,
                       default=2048,
                       help='FFT size')
    
    parser.add_argument('--source', 
                       choices=['simulation', 'rtlsdr', 'hackrf'],
                       default='simulation',
                       help='Signal source type')
    
    parser.add_argument('--device', 
                       type=str,
                       default='',
                       help='Device arguments')
    
    parser.add_argument('-g', '--gain', 
                       type=float,
                       default=30,
                       help='RF gain in dB')
    
    parser.add_argument('--averaging', 
                       type=float,
                       default=0.8,
                       help='Spectrum averaging (0-1)')
    
    parser.add_argument('--sweep', 
                       action='store_true',
                       help='Enable frequency sweeping')
    
    parser.add_argument('--plot', 
                       action='store_true',
                       help='Enable real-time plotting')
    
    parser.add_argument('--threshold', 
                       type=float,
                       default=-60,
                       help='Peak detection threshold (dB)')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Convert frequencies to Hz
    start_freq = args.start_freq * 1e6
    stop_freq = args.stop_freq * 1e6
    
    print("GNU Radio Spectrum Analyzer")
    print("=" * 50)
    print(f"Frequency Range: {args.start_freq} - {args.stop_freq} MHz")
    print(f"Sample Rate:     {args.sample_rate/1e6} MHz")
    print(f"FFT Size:        {args.fft_size}")
    print(f"Source:          {args.source}")
    print(f"Gain:            {args.gain} dB")
    print(f"Averaging:       {args.averaging}")
    print(f"Threshold:       {args.threshold} dB")
    print("=" * 50)
    
    # Create analyzer
    analyzer = SpectrumAnalyzer(
        start_freq=start_freq,
        stop_freq=stop_freq,
        sample_rate=args.sample_rate,
        fft_size=args.fft_size,
        source_type=args.source,
        device_args=args.device,
        gain=args.gain,
        averaging=args.averaging
    )
    
    analyzer.threshold = args.threshold
    
    # Start analyzer
    analyzer.start()
    
    if args.sweep:
        # Start frequency sweep
        analyzer.start_sweep()
    
    if args.plot:
        # Real-time plotting
        print("\nStarting real-time plot...")
        plotter = SpectrumPlotter(analyzer)
        plotter.start()
    else:
        # Console mode
        print("\nAnalyzing spectrum...")
        print("Commands:")
        print("  f <freq>  - Set center frequency (MHz)")
        print("  g <gain>  - Set gain (dB)")
        print("  s         - Start/stop sweep")
        print("  p         - Show peaks")
        print("  q         - Quit")
        print()
        
        try:
            sweeping = args.sweep
            while True:
                cmd = input("> ").strip().lower()
                
                if cmd.startswith('f '):
                    freq = float(cmd.split()[1]) * 1e6
                    analyzer.set_center_freq(freq)
                    print(f"Tuned to {freq/1e6:.1f} MHz")
                    
                elif cmd.startswith('g '):
                    gain = float(cmd.split()[1])
                    analyzer.set_gain(gain)
                    print(f"Gain set to {gain} dB")
                    
                elif cmd == 's':
                    if sweeping:
                        analyzer.stop_sweep()
                        sweeping = False
                        print("Sweep stopped")
                    else:
                        analyzer.start_sweep()
                        sweeping = True
                        print("Sweep started")
                    
                elif cmd == 'p':
                    peaks = analyzer.find_peaks()
                    print(f"\nFound {len(peaks)} peaks:")
                    for freq, power in sorted(peaks)[:10]:
                        print(f"  {freq/1e6:8.3f} MHz: {power:6.1f} dB")
                    
                elif cmd == 'q':
                    break
                    
        except (KeyboardInterrupt, EOFError):
            pass
        
        # Stop sweep if running
        if sweeping:
            analyzer.stop_sweep()
    
    # Stop analyzer
    print("\nShutting down...")
    analyzer.stop()
    analyzer.wait()
    
    # Print summary
    if len(analyzer.peak_frequencies) > 0:
        print("\nSignal Summary:")
        print("-" * 40)
        
        # Find most common frequencies
        freq_bins, counts = np.unique(
            np.round(np.array(analyzer.peak_frequencies) / 1e6, 1),
            return_counts=True)
        
        # Sort by count
        sorted_idx = np.argsort(counts)[::-1]
        
        print("Top detected frequencies:")
        for i in range(min(10, len(sorted_idx))):
            idx = sorted_idx[i]
            print(f"  {freq_bins[idx]:8.1f} MHz: {counts[idx]:3d} detections")
    
    print("\nDone!")


if __name__ == '__main__':
    main()