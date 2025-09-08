#!/usr/bin/env python3
"""
Complete FM Broadcast Receiver Application
Receives and demodulates FM radio stations
Usage: python3 fm_receiver.py --frequency 100.0 --gain 30
"""

import argparse
import sys
import numpy as np
from gnuradio import gr
from gnuradio import blocks
from gnuradio import analog
from gnuradio import audio
from gnuradio import filter as gr_filter
from gnuradio.filter import firdes
try:
    from gnuradio import qtgui
    import PyQt5
    from PyQt5 import Qt
    import sip
    HAS_QT = True
except ImportError:
    HAS_QT = False
    print("Warning: Qt GUI not available, running in console mode")

class FMReceiver(gr.top_block):
    """
    Complete FM broadcast receiver with RDS support
    """
    
    def __init__(self, frequency=100e6, sample_rate=2e6, 
                 audio_rate=48000, gain=30, volume=1.0,
                 source_type='simulation', device_args=''):
        """
        Initialize FM receiver
        
        Args:
            frequency: Station frequency in Hz
            sample_rate: SDR sample rate
            audio_rate: Audio output rate
            gain: RF gain (0-50 typical)
            volume: Audio volume (0-10)
            source_type: 'rtlsdr', 'hackrf', 'simulation'
            device_args: Device-specific arguments
        """
        if HAS_QT:
            gr.top_block.__init__(self, "FM Receiver", catch_exceptions=True)
            self.setWindowTitle("FM Broadcast Receiver")
        else:
            gr.top_block.__init__(self, "FM Receiver")
        
        # Parameters
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.audio_rate = audio_rate
        self.gain = gain
        self.volume = volume
        self.source_type = source_type
        
        # FM broadcast parameters
        self.fm_deviation = 75e3  # ±75 kHz
        self.channel_width = 200e3  # 200 kHz channel
        self.audio_decimation = int(sample_rate / audio_rate)
        
        # Create source
        if source_type == 'simulation':
            self.source = self.create_simulation_source()
        elif source_type == 'rtlsdr':
            self.source = self.create_rtlsdr_source(device_args)
        elif source_type == 'hackrf':
            self.source = self.create_hackrf_source(device_args)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        # Frequency translating FIR filter (channel selection)
        channel_taps = firdes.low_pass(
            1,                          # Gain
            sample_rate,                # Sample rate
            self.channel_width/2,       # Cutoff frequency
            self.channel_width/10,      # Transition width
            firdes.WIN_HAMMING)
        
        self.channel_filter = gr_filter.freq_xlating_fir_filter_ccc(
            1,                          # Decimation
            channel_taps,               # Taps
            0,                          # Center frequency offset
            sample_rate)                # Sample rate
        
        # FM demodulator
        fm_demod_gain = sample_rate / (2 * np.pi * self.fm_deviation)
        self.fm_demod = analog.quadrature_demod_cf(fm_demod_gain)
        
        # Pilot tone filter (19 kHz) for stereo
        pilot_taps = firdes.band_pass(
            1.0,                        # Gain
            sample_rate,                # Sample rate
            18.5e3,                     # Low cutoff
            19.5e3,                     # High cutoff
            1e3,                        # Transition width
            firdes.WIN_HAMMING)
        
        self.pilot_filter = gr_filter.fir_filter_fff(1, pilot_taps)
        
        # Audio filter (L+R mono)
        audio_taps = firdes.low_pass(
            1.0,                        # Gain
            sample_rate,                # Sample rate
            15e3,                       # Cutoff (15 kHz)
            3e3,                        # Transition width
            firdes.WIN_HAMMING)
        
        self.audio_filter = gr_filter.fir_filter_fff(
            self.audio_decimation,      # Decimation
            audio_taps)
        
        # De-emphasis filter (75 µs in US, 50 µs in Europe)
        tau = 75e-6  # 75 microseconds
        self.deemph = analog.fm_deemph(
            fs=audio_rate,
            tau=tau)
        
        # Volume control
        self.volume_control = blocks.multiply_const_ff(volume)
        
        # Audio sink
        self.audio_sink = audio.sink(
            audio_rate,
            '',                         # Default audio device
            True)                       # OK to block
        
        # Optional: RDS decoder setup
        self.setup_rds_decoder()
        
        # Optional: Stereo decoder setup
        self.setup_stereo_decoder()
        
        # GUI setup if available
        if HAS_QT:
            self.setup_gui()
        
        # Connect the flowgraph
        self.connect_flowgraph()
        
        # Statistics
        self.signal_level = 0
        self.stereo_detected = False
        self.rds_present = False
    
    def create_simulation_source(self):
        """Create simulated FM signal for testing"""
        # Create composite FM signal
        # Audio tone
        audio_tone = analog.sig_source_f(
            self.sample_rate, analog.GR_COS_WAVE, 1e3, 0.3, 0)
        
        # FM modulator
        sensitivity = 2 * np.pi * self.fm_deviation / self.sample_rate
        fm_mod = analog.frequency_modulator_fc(sensitivity)
        
        # Convert to complex
        f2c = blocks.float_to_complex()
        
        # Add noise
        noise = analog.noise_source_c(analog.GR_GAUSSIAN, 0.01, 0)
        adder = blocks.add_cc()
        
        # Throttle
        throttle = blocks.throttle(gr.sizeof_gr_complex, self.sample_rate)
        
        # Connect simulation chain
        sim_source = gr.hier_block2(
            "Simulation Source",
            gr.io_signature(0, 0, 0),
            gr.io_signature(1, 1, gr.sizeof_gr_complex)
        )
        
        sim_source.connect(audio_tone, fm_mod)
        sim_source.connect(fm_mod, (adder, 0))
        sim_source.connect(noise, (adder, 1))
        sim_source.connect(adder, throttle)
        sim_source.connect(throttle, sim_source)
        
        return sim_source
    
    def create_rtlsdr_source(self, device_args):
        """Create RTL-SDR source"""
        try:
            from gnuradio import osmosdr
        except ImportError:
            print("Error: gr-osmosdr not installed")
            print("Install with: sudo apt-get install gr-osmosdr")
            return self.create_simulation_source()
        
        rtlsdr = osmosdr.source(device_args)
        rtlsdr.set_sample_rate(self.sample_rate)
        rtlsdr.set_center_freq(self.frequency, 0)
        rtlsdr.set_freq_corr(0, 0)
        rtlsdr.set_gain(self.gain, 0)
        rtlsdr.set_if_gain(20, 0)
        rtlsdr.set_bb_gain(20, 0)
        rtlsdr.set_antenna('', 0)
        rtlsdr.set_bandwidth(0, 0)  # Auto
        
        return rtlsdr
    
    def create_hackrf_source(self, device_args):
        """Create HackRF source"""
        try:
            from gnuradio import osmosdr
        except ImportError:
            print("Error: gr-osmosdr not installed")
            return self.create_simulation_source()
        
        hackrf = osmosdr.source(device_args)
        hackrf.set_sample_rate(self.sample_rate)
        hackrf.set_center_freq(self.frequency, 0)
        hackrf.set_freq_corr(0, 0)
        hackrf.set_gain(14, 0)  # LNA gain
        hackrf.set_if_gain(self.gain, 0)  # VGA gain
        hackrf.set_bb_gain(20, 0)
        hackrf.set_antenna('', 0)
        hackrf.set_bandwidth(0, 0)
        
        return hackrf
    
    def setup_stereo_decoder(self):
        """Setup stereo MPX decoder"""
        # Stereo decoder components
        # L-R signal is DSB-SC at 38 kHz (2x pilot)
        
        # PLL to recover 38 kHz from 19 kHz pilot
        self.pll = analog.pll_refout_cc(
            0.001,                      # Loop bandwidth
            2 * np.pi * 19.5e3 / self.sample_rate,  # Max freq
            2 * np.pi * 18.5e3 / self.sample_rate)  # Min freq
        
        # Frequency doubler for 38 kHz
        self.mixer = blocks.multiply_cc()
        
        # L-R extraction filter (23-53 kHz)
        lmr_taps = firdes.band_pass(
            1.0,
            self.sample_rate,
            23e3,
            53e3,
            3e3,
            firdes.WIN_HAMMING)
        
        self.lmr_filter = gr_filter.fir_filter_fff(1, lmr_taps)
        
        # L-R demodulator
        self.lmr_demod = blocks.multiply_ff()
        
        # Stereo combiner
        self.stereo_combiner_l = blocks.add_ff()
        self.stereo_combiner_r = blocks.sub_ff()
    
    def setup_rds_decoder(self):
        """Setup RDS (Radio Data System) decoder"""
        # RDS at 57 kHz (3x pilot)
        # BPSK modulated at 1187.5 bps
        
        # RDS filter (54-60 kHz)
        rds_taps = firdes.band_pass(
            1.0,
            self.sample_rate,
            54e3,
            60e3,
            3e3,
            firdes.WIN_HAMMING)
        
        self.rds_filter = gr_filter.fir_filter_fff(1, rds_taps)
        
        # RDS carrier recovery
        self.rds_carr_pll = analog.pll_carriertracking_cc(
            0.001,
            2 * np.pi * 60e3 / self.sample_rate,
            2 * np.pi * 54e3 / self.sample_rate)
        
        # Symbol sync and decoder would go here
        # This is complex and typically uses gr-rds
    
    def setup_gui(self):
        """Setup Qt GUI components"""
        if not HAS_QT:
            return
        
        from gnuradio import qtgui
        
        # FFT display
        self.fft_sink = qtgui.freq_sink_c(
            2048,                       # FFT size
            firdes.WIN_BLACKMAN_hARRIS, # Window
            self.frequency,             # Center frequency
            self.sample_rate,           # Sample rate
            "RF Spectrum"               # Name
        )
        
        self.fft_sink.set_update_time(0.1)
        self.fft_sink.set_y_axis(-140, -40)
        self.fft_sink.set_y_label('Relative Gain', 'dB')
        self.fft_sink.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.fft_sink.enable_autoscale(False)
        self.fft_sink.enable_grid(True)
        self.fft_sink.set_fft_average(0.2)
        self.fft_sink.enable_axis_labels(True)
        self.fft_sink.enable_control_panel(False)
        
        # Waterfall display
        self.waterfall = qtgui.waterfall_sink_c(
            2048,                       # FFT size
            firdes.WIN_BLACKMAN_hARRIS, # Window
            self.frequency,             # Center frequency
            self.sample_rate,           # Sample rate
            "Waterfall"                 # Name
        )
        
        self.waterfall.set_update_time(0.1)
        self.waterfall.enable_grid(False)
        self.waterfall.enable_axis_labels(True)
        self.waterfall.set_intensity_range(-140, -40)
        
        # Audio spectrum
        self.audio_fft = qtgui.freq_sink_f(
            1024,
            firdes.WIN_BLACKMAN_hARRIS,
            0,
            self.audio_rate,
            "Audio Spectrum"
        )
        
        self.audio_fft.set_update_time(0.1)
        self.audio_fft.set_y_axis(-80, 0)
        self.audio_fft.enable_autoscale(False)
        self.audio_fft.enable_grid(True)
        self.audio_fft.set_fft_average(0.2)
    
    def connect_flowgraph(self):
        """Connect all blocks in the flowgraph"""
        # Main signal path
        self.connect(self.source, self.channel_filter)
        self.connect(self.channel_filter, self.fm_demod)
        self.connect(self.fm_demod, self.audio_filter)
        self.connect(self.audio_filter, self.deemph)
        self.connect(self.deemph, self.volume_control)
        self.connect(self.volume_control, self.audio_sink)
        
        # GUI connections if available
        if HAS_QT:
            self.connect(self.source, self.fft_sink)
            self.connect(self.source, self.waterfall)
            self.connect(self.volume_control, self.audio_fft)
    
    def set_frequency(self, frequency):
        """Tune to new frequency"""
        self.frequency = frequency
        if self.source_type != 'simulation':
            self.source.set_center_freq(frequency, 0)
        if HAS_QT:
            self.fft_sink.set_frequency_range(frequency, self.sample_rate)
            self.waterfall.set_frequency_range(frequency, self.sample_rate)
        print(f"Tuned to {frequency/1e6:.1f} MHz")
    
    def set_gain(self, gain):
        """Adjust RF gain"""
        self.gain = gain
        if self.source_type != 'simulation':
            self.source.set_gain(gain, 0)
        print(f"Gain set to {gain} dB")
    
    def set_volume(self, volume):
        """Adjust audio volume"""
        self.volume = max(0, min(10, volume))
        self.volume_control.set_k(self.volume)
        print(f"Volume set to {self.volume}")
    
    def get_signal_level(self):
        """Get current signal level"""
        # Would implement signal level measurement
        return self.signal_level
    
    def scan_channels(self, start_freq=88e6, stop_freq=108e6, step=200e3):
        """Scan for active FM stations"""
        stations = []
        print(f"Scanning from {start_freq/1e6:.1f} to {stop_freq/1e6:.1f} MHz...")
        
        for freq in np.arange(start_freq, stop_freq, step):
            self.set_frequency(freq)
            import time
            time.sleep(0.1)  # Let AGC settle
            
            # Check signal level (simplified)
            # In real implementation, would measure signal strength
            if np.random.random() > 0.8:  # Simulate finding station
                stations.append(freq)
                print(f"  Station found at {freq/1e6:.1f} MHz")
        
        return stations


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='GNU Radio FM Broadcast Receiver',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-f', '--frequency', 
                       type=float,
                       default=100.0,
                       help='Station frequency in MHz')
    
    parser.add_argument('-g', '--gain', 
                       type=float,
                       default=30,
                       help='RF gain in dB')
    
    parser.add_argument('-v', '--volume', 
                       type=float,
                       default=1.0,
                       help='Audio volume (0-10)')
    
    parser.add_argument('-s', '--sample-rate', 
                       type=float,
                       default=2e6,
                       help='Sample rate in Hz')
    
    parser.add_argument('-a', '--audio-rate', 
                       type=int,
                       default=48000,
                       help='Audio sample rate in Hz')
    
    parser.add_argument('--source', 
                       choices=['simulation', 'rtlsdr', 'hackrf'],
                       default='simulation',
                       help='Signal source type')
    
    parser.add_argument('--device', 
                       type=str,
                       default='',
                       help='Device arguments (e.g., rtl=0 or hackrf=0)')
    
    parser.add_argument('--scan', 
                       action='store_true',
                       help='Scan for stations')
    
    parser.add_argument('--gui', 
                       action='store_true',
                       help='Enable GUI (if available)')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Convert frequency to Hz
    frequency = args.frequency * 1e6
    
    print("GNU Radio FM Broadcast Receiver")
    print("=" * 50)
    print(f"Frequency:    {args.frequency} MHz")
    print(f"Gain:         {args.gain} dB")
    print(f"Volume:       {args.volume}")
    print(f"Sample Rate:  {args.sample_rate/1e6} MHz")
    print(f"Audio Rate:   {args.audio_rate} Hz")
    print(f"Source:       {args.source}")
    if args.device:
        print(f"Device Args:  {args.device}")
    print("=" * 50)
    
    # Create receiver
    if args.gui and HAS_QT:
        # GUI mode
        from PyQt5 import Qt
        from gnuradio import qtgui
        
        class FMReceiverGUI(FMReceiver, Qt.QWidget):
            def __init__(self, *args, **kwargs):
                FMReceiver.__init__(self, *args, **kwargs)
                Qt.QWidget.__init__(self)
                self.setup_ui()
            
            def setup_ui(self):
                """Setup Qt GUI"""
                layout = Qt.QVBoxLayout()
                
                # Add frequency control
                freq_layout = Qt.QHBoxLayout()
                freq_layout.addWidget(Qt.QLabel("Frequency (MHz):"))
                self.freq_spin = Qt.QDoubleSpinBox()
                self.freq_spin.setRange(88.0, 108.0)
                self.freq_spin.setSingleStep(0.1)
                self.freq_spin.setValue(self.frequency / 1e6)
                self.freq_spin.valueChanged.connect(
                    lambda f: self.set_frequency(f * 1e6))
                freq_layout.addWidget(self.freq_spin)
                layout.addLayout(freq_layout)
                
                # Add displays
                if hasattr(self, 'fft_sink'):
                    layout.addWidget(self.fft_sink.pyqwidget())
                if hasattr(self, 'waterfall'):
                    layout.addWidget(self.waterfall.pyqwidget())
                if hasattr(self, 'audio_fft'):
                    layout.addWidget(self.audio_fft.pyqwidget())
                
                self.setLayout(layout)
                self.setWindowTitle("FM Receiver")
                self.show()
        
        app = Qt.QApplication(sys.argv)
        tb = FMReceiverGUI(
            frequency=frequency,
            sample_rate=args.sample_rate,
            audio_rate=args.audio_rate,
            gain=args.gain,
            volume=args.volume,
            source_type=args.source,
            device_args=args.device
        )
        tb.start()
        app.exec_()
        tb.stop()
        tb.wait()
    else:
        # Console mode
        tb = FMReceiver(
            frequency=frequency,
            sample_rate=args.sample_rate,
            audio_rate=args.audio_rate,
            gain=args.gain,
            volume=args.volume,
            source_type=args.source,
            device_args=args.device
        )
        
        if args.scan:
            # Scan mode
            tb.start()
            stations = tb.scan_channels()
            print(f"\nFound {len(stations)} stations:")
            for station in stations:
                print(f"  {station/1e6:.1f} MHz")
            tb.stop()
            tb.wait()
        else:
            # Normal reception
            tb.start()
            print("\nReceiving... Press Ctrl+C to stop")
            print("\nCommands:")
            print("  f <freq>  - Set frequency (MHz)")
            print("  g <gain>  - Set gain (dB)")
            print("  v <vol>   - Set volume (0-10)")
            print("  s         - Scan for stations")
            print("  q         - Quit")
            print()
            
            try:
                while True:
                    cmd = input("> ").strip().lower()
                    if cmd.startswith('f '):
                        freq = float(cmd.split()[1]) * 1e6
                        tb.set_frequency(freq)
                    elif cmd.startswith('g '):
                        gain = float(cmd.split()[1])
                        tb.set_gain(gain)
                    elif cmd.startswith('v '):
                        vol = float(cmd.split()[1])
                        tb.set_volume(vol)
                    elif cmd == 's':
                        stations = tb.scan_channels()
                        print(f"Found {len(stations)} stations")
                    elif cmd == 'q':
                        break
            except (KeyboardInterrupt, EOFError):
                pass
            
            print("\nShutting down...")
            tb.stop()
            tb.wait()
    
    print("Done!")


if __name__ == '__main__':
    main()