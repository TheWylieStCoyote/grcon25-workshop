#!/usr/bin/env python3
"""
AM Transmitter Application
Amplitude modulation transmitter with audio input
Usage: python3 am_transmitter.py --frequency 1.0 --modulation 0.8
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
import wave
import time

class AMTransmitter(gr.top_block):
    """
    Complete AM transmitter with various modulation modes
    """
    
    def __init__(self, carrier_freq=1e6, sample_rate=2e6, 
                 audio_rate=48000, modulation_index=0.8,
                 mode='AM', audio_source='mic', audio_file=None):
        """
        Initialize AM transmitter
        
        Args:
            carrier_freq: Carrier frequency in Hz
            sample_rate: Output sample rate
            audio_rate: Audio input rate
            modulation_index: Modulation depth (0-1)
            mode: 'AM', 'DSB', 'SSB-USB', 'SSB-LSB'
            audio_source: 'mic', 'file', 'tone'
            audio_file: Path to audio file if source is 'file'
        """
        gr.top_block.__init__(self, "AM Transmitter")
        
        # Parameters
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        self.audio_rate = audio_rate
        self.modulation_index = modulation_index
        self.mode = mode
        self.audio_source_type = audio_source
        
        # Create audio source
        if audio_source == 'mic':
            self.audio_source = audio.source(audio_rate, '', True)
        elif audio_source == 'file':
            if audio_file:
                self.audio_source = self.create_file_source(audio_file)
            else:
                print("No audio file specified, using tone generator")
                self.audio_source = self.create_tone_source()
        else:  # tone
            self.audio_source = self.create_tone_source()
        
        # Audio processing
        # Pre-emphasis filter (optional, for broadcast)
        tau = 75e-6  # 75 microseconds
        self.preemph = analog.fm_preemph(
            fs=audio_rate,
            tau=tau,
            fh=-1.0)  # No high freq limit
        
        # Audio filter (limit bandwidth)
        audio_taps = firdes.low_pass(
            1.0,                    # Gain
            audio_rate,             # Sample rate
            5000,                   # Cutoff frequency (5 kHz for AM)
            500,                    # Transition width
            firdes.WIN_HAMMING)
        
        self.audio_filter = gr_filter.fir_filter_fff(1, audio_taps)
        
        # AGC for consistent modulation
        self.audio_agc = analog.agc_ff(
            1e-3,                   # Attack rate
            1.0,                    # Reference
            1.0)                    # Initial gain
        self.audio_agc.set_max_gain(10.0)
        
        # Resampler to match output rate
        self.resampler = gr_filter.rational_resampler_fff(
            interpolation=int(sample_rate),
            decimation=int(audio_rate),
            taps=None,
            fractional_bw=None)
        
        # Modulation index control
        self.mod_index_mult = blocks.multiply_const_ff(modulation_index)
        
        # DC offset for AM (carrier level)
        self.dc_offset = blocks.add_const_ff(1.0)
        
        # Create modulator based on mode
        if mode == 'AM':
            self.modulator = self.create_am_modulator()
        elif mode == 'DSB':
            self.modulator = self.create_dsb_modulator()
        elif mode == 'SSB-USB':
            self.modulator = self.create_ssb_modulator(upper=True)
        elif mode == 'SSB-LSB':
            self.modulator = self.create_ssb_modulator(upper=False)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Output processing
        # Bandpass filter to clean up signal
        if mode in ['SSB-USB', 'SSB-LSB']:
            # SSB needs different filtering
            if mode == 'SSB-USB':
                bp_low = 300
                bp_high = 3000
            else:
                bp_low = -3000
                bp_high = -300
        else:
            bp_low = -5000
            bp_high = 5000
        
        bp_taps = firdes.band_pass(
            1.0,                    # Gain
            sample_rate,            # Sample rate
            bp_low,                 # Low cutoff
            bp_high,                # High cutoff
            500,                    # Transition width
            firdes.WIN_HAMMING)
        
        self.output_filter = gr_filter.fir_filter_ccc(1, bp_taps)
        
        # Output options
        self.setup_output_sinks()
        
        # Connect the flowgraph
        self.connect_flowgraph()
        
        # Statistics
        self.tx_power = 0
        self.peak_deviation = 0
    
    def create_tone_source(self):
        """Create test tone generator"""
        # Two-tone test signal
        tone1 = analog.sig_source_f(
            self.audio_rate, analog.GR_COS_WAVE, 1000, 0.4, 0)
        tone2 = analog.sig_source_f(
            self.audio_rate, analog.GR_COS_WAVE, 2000, 0.3, 0)
        adder = blocks.add_ff()
        
        # Create hierarchical block
        tone_source = gr.hier_block2(
            "Tone Source",
            gr.io_signature(0, 0, 0),
            gr.io_signature(1, 1, gr.sizeof_float)
        )
        
        tone_source.connect(tone1, (adder, 0))
        tone_source.connect(tone2, (adder, 1))
        tone_source.connect(adder, tone_source)
        
        return tone_source
    
    def create_file_source(self, filename):
        """Create audio file source"""
        # Check if file exists
        import os
        if not os.path.exists(filename):
            print(f"Audio file not found: {filename}")
            return self.create_tone_source()
        
        # For WAV files
        if filename.endswith('.wav'):
            file_source = blocks.wavfile_source(filename, True)
        else:
            # For raw audio files
            file_source = blocks.file_source(
                gr.sizeof_float, filename, True)
        
        return file_source
    
    def create_am_modulator(self):
        """Create standard AM modulator"""
        # AM: output = (1 + m*audio) * carrier
        carrier_source = analog.sig_source_c(
            self.sample_rate,
            analog.GR_COS_WAVE,
            0,  # IF frequency (0 for baseband)
            1.0,
            0)
        
        # Float to complex
        f2c = blocks.float_to_complex()
        
        # Multiply with carrier
        mixer = blocks.multiply_cc()
        
        # Create hierarchical block
        am_mod = gr.hier_block2(
            "AM Modulator",
            gr.io_signature(1, 1, gr.sizeof_float),
            gr.io_signature(1, 1, gr.sizeof_gr_complex)
        )
        
        am_mod.connect(am_mod, (f2c, 0))
        am_mod.connect(blocks.null_source(gr.sizeof_float), (f2c, 1))
        am_mod.connect(f2c, (mixer, 0))
        am_mod.connect(carrier_source, (mixer, 1))
        am_mod.connect(mixer, am_mod)
        
        return am_mod
    
    def create_dsb_modulator(self):
        """Create DSB-SC (Double Sideband Suppressed Carrier) modulator"""
        # DSB: output = audio * carrier (no DC offset)
        carrier_source = analog.sig_source_c(
            self.sample_rate,
            analog.GR_COS_WAVE,
            0,
            1.0,
            0)
        
        f2c = blocks.float_to_complex()
        mixer = blocks.multiply_cc()
        
        dsb_mod = gr.hier_block2(
            "DSB Modulator",
            gr.io_signature(1, 1, gr.sizeof_float),
            gr.io_signature(1, 1, gr.sizeof_gr_complex)
        )
        
        # Note: No DC offset for DSB-SC
        dsb_mod.connect(dsb_mod, (f2c, 0))
        dsb_mod.connect(blocks.null_source(gr.sizeof_float), (f2c, 1))
        dsb_mod.connect(f2c, (mixer, 0))
        dsb_mod.connect(carrier_source, (mixer, 1))
        dsb_mod.connect(mixer, dsb_mod)
        
        return dsb_mod
    
    def create_ssb_modulator(self, upper=True):
        """Create SSB modulator using filter method"""
        # Hilbert transform for SSB generation
        hilbert_taps = firdes.hilbert(65)
        hilbert = gr_filter.fir_filter_fff(1, hilbert_taps)
        
        # Delay to compensate for Hilbert filter
        delay = blocks.delay(gr.sizeof_float, (len(hilbert_taps)-1)//2)
        
        # Create I and Q components
        f2c = blocks.float_to_complex()
        
        # Carrier sources (90 degree phase shift)
        if upper:
            # USB: I*cos - Q*sin
            carrier_i = analog.sig_source_c(
                self.sample_rate, analog.GR_COS_WAVE, 0, 1.0, 0)
        else:
            # LSB: I*cos + Q*sin
            carrier_i = analog.sig_source_c(
                self.sample_rate, analog.GR_COS_WAVE, 0, 1.0, 0)
        
        mixer = blocks.multiply_cc()
        
        # Create hierarchical block
        ssb_mod = gr.hier_block2(
            f"SSB {'USB' if upper else 'LSB'} Modulator",
            gr.io_signature(1, 1, gr.sizeof_float),
            gr.io_signature(1, 1, gr.sizeof_gr_complex)
        )
        
        # Connect Hilbert transform path
        ssb_mod.connect(ssb_mod, hilbert, (f2c, 1))  # Q component
        ssb_mod.connect(ssb_mod, delay, (f2c, 0))    # I component
        
        # Mix with carrier
        ssb_mod.connect(f2c, (mixer, 0))
        ssb_mod.connect(carrier_i, (mixer, 1))
        
        # Apply sideband selection
        if not upper:
            # For LSB, conjugate the result
            conj = blocks.conjugate_cc()
            ssb_mod.connect(mixer, conj, ssb_mod)
        else:
            ssb_mod.connect(mixer, ssb_mod)
        
        return ssb_mod
    
    def setup_output_sinks(self):
        """Setup various output options"""
        # File sink for recording
        self.iq_file_sink = blocks.file_sink(
            gr.sizeof_gr_complex,
            '/tmp/am_transmit.iq',
            False)
        self.iq_file_sink.set_unbuffered(False)
        
        # Null sink for testing
        self.null_sink = blocks.null_sink(gr.sizeof_gr_complex)
        
        # For SDR hardware output (would add USRP/HackRF here)
        self.hardware_sink = None
        
        # Throttle for simulation
        self.throttle = blocks.throttle(
            gr.sizeof_gr_complex,
            self.sample_rate)
    
    def connect_flowgraph(self):
        """Connect all blocks in the flowgraph"""
        # Audio processing chain
        self.connect(self.audio_source, self.audio_filter)
        self.connect(self.audio_filter, self.audio_agc)
        
        # Don't use pre-emphasis for AM (it's for FM)
        # self.connect(self.audio_agc, self.preemph)
        # self.connect(self.preemph, self.resampler)
        
        self.connect(self.audio_agc, self.resampler)
        self.connect(self.resampler, self.mod_index_mult)
        
        # Add DC offset for standard AM
        if self.mode == 'AM':
            self.connect(self.mod_index_mult, self.dc_offset)
            self.connect(self.dc_offset, self.modulator)
        else:
            self.connect(self.mod_index_mult, self.modulator)
        
        # Output processing
        self.connect(self.modulator, self.output_filter)
        self.connect(self.output_filter, self.throttle)
        
        # Connect to outputs
        self.connect(self.throttle, self.iq_file_sink)
        self.connect(self.throttle, self.null_sink)
    
    def set_modulation_index(self, index):
        """Adjust modulation index"""
        self.modulation_index = max(0, min(1, index))
        self.mod_index_mult.set_k(self.modulation_index)
        print(f"Modulation index set to {self.modulation_index:.2f}")
    
    def set_carrier_freq(self, freq):
        """Set carrier frequency (for hardware output)"""
        self.carrier_freq = freq
        print(f"Carrier frequency set to {freq/1e6:.3f} MHz")
    
    def get_power_level(self):
        """Get current transmit power level"""
        return self.tx_power
    
    def enable_compression(self, ratio=2.0, threshold=0.7):
        """Enable audio compression"""
        # Would implement audio compressor here
        print(f"Audio compression enabled: {ratio}:1 ratio")


def analyze_modulation(filename='/tmp/am_transmit.iq', 
                       sample_rate=2e6, duration=1.0):
    """Analyze the transmitted signal"""
    import matplotlib.pyplot as plt
    from scipy import signal
    
    # Read IQ data
    try:
        data = np.fromfile(filename, dtype=np.complex64, 
                          count=int(sample_rate * duration))
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return
    
    if len(data) == 0:
        print("No data to analyze")
        return
    
    # Time domain analysis
    time = np.arange(len(data)) / sample_rate
    envelope = np.abs(data)
    phase = np.angle(data)
    
    # Frequency domain analysis
    freqs, psd = signal.welch(data, sample_rate, nperseg=2048)
    
    # Modulation depth measurement
    carrier_idx = np.argmax(np.abs(np.fft.fft(data[:8192])))
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Time domain - envelope
    axes[0, 0].plot(time[:1000], envelope[:1000])
    axes[0, 0].set_title('Envelope (Time Domain)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True)
    
    # Time domain - IQ
    axes[0, 1].plot(time[:1000], np.real(data[:1000]), 'b', 
                   label='I', alpha=0.7)
    axes[0, 1].plot(time[:1000], np.imag(data[:1000]), 'r', 
                   label='Q', alpha=0.7)
    axes[0, 1].set_title('I/Q Components')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Frequency domain
    axes[1, 0].semilogy(freqs/1e3, psd)
    axes[1, 0].set_title('Power Spectral Density')
    axes[1, 0].set_xlabel('Frequency (kHz)')
    axes[1, 0].set_ylabel('PSD')
    axes[1, 0].grid(True)
    
    # Constellation diagram
    axes[1, 1].scatter(np.real(data[::10]), np.imag(data[::10]), 
                      alpha=0.3, s=1)
    axes[1, 1].set_title('IQ Constellation')
    axes[1, 1].set_xlabel('I')
    axes[1, 1].set_ylabel('Q')
    axes[1, 1].grid(True)
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics
    print("\nModulation Analysis:")
    print("-" * 40)
    print(f"Peak amplitude: {np.max(envelope):.3f}")
    print(f"Average power: {np.mean(envelope**2):.3f}")
    print(f"Peak-to-average ratio: {np.max(envelope)/np.mean(envelope):.2f}")
    
    # Estimate modulation index for AM
    if np.mean(envelope) > 0:
        mod_index = (np.max(envelope) - np.min(envelope)) / (
                   np.max(envelope) + np.min(envelope))
        print(f"Estimated modulation index: {mod_index:.2f}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='GNU Radio AM Transmitter',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-f', '--frequency', 
                       type=float,
                       default=1.0,
                       help='Carrier frequency in MHz')
    
    parser.add_argument('-m', '--modulation', 
                       type=float,
                       default=0.8,
                       help='Modulation index (0-1)')
    
    parser.add_argument('--mode', 
                       choices=['AM', 'DSB', 'SSB-USB', 'SSB-LSB'],
                       default='AM',
                       help='Modulation mode')
    
    parser.add_argument('-s', '--sample-rate', 
                       type=float,
                       default=2e6,
                       help='Sample rate in Hz')
    
    parser.add_argument('-a', '--audio-rate', 
                       type=int,
                       default=48000,
                       help='Audio sample rate in Hz')
    
    parser.add_argument('--source', 
                       choices=['mic', 'file', 'tone'],
                       default='tone',
                       help='Audio source')
    
    parser.add_argument('--file', 
                       type=str,
                       help='Audio file path (if source is file)')
    
    parser.add_argument('--duration', 
                       type=float,
                       default=10,
                       help='Transmission duration in seconds')
    
    parser.add_argument('--analyze', 
                       action='store_true',
                       help='Analyze transmitted signal')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Convert frequency to Hz
    carrier_freq = args.frequency * 1e6
    
    print("GNU Radio AM Transmitter")
    print("=" * 50)
    print(f"Mode:         {args.mode}")
    print(f"Carrier:      {args.frequency} MHz")
    print(f"Modulation:   {args.modulation * 100:.0f}%")
    print(f"Sample Rate:  {args.sample_rate/1e6} MHz")
    print(f"Audio Rate:   {args.audio_rate} Hz")
    print(f"Audio Source: {args.source}")
    if args.file:
        print(f"Audio File:   {args.file}")
    print(f"Output File:  /tmp/am_transmit.iq")
    print("=" * 50)
    
    # Create transmitter
    tx = AMTransmitter(
        carrier_freq=carrier_freq,
        sample_rate=args.sample_rate,
        audio_rate=args.audio_rate,
        modulation_index=args.modulation,
        mode=args.mode,
        audio_source=args.source,
        audio_file=args.file
    )
    
    # Start transmission
    print(f"\nTransmitting for {args.duration} seconds...")
    print("Press Ctrl+C to stop early")
    
    tx.start()
    
    try:
        # Interactive mode
        if args.source == 'mic':
            print("\nTransmitting from microphone...")
            print("Commands:")
            print("  m <value>  - Set modulation index (0-1)")
            print("  q          - Quit")
            print()
            
            import select
            import sys
            
            while True:
                # Check for input with timeout
                ready = select.select([sys.stdin], [], [], 1)[0]
                if ready:
                    cmd = sys.stdin.readline().strip().lower()
                    if cmd.startswith('m '):
                        mod = float(cmd.split()[1])
                        tx.set_modulation_index(mod)
                    elif cmd == 'q':
                        break
        else:
            # Fixed duration
            time.sleep(args.duration)
    
    except KeyboardInterrupt:
        print("\nTransmission interrupted")
    
    tx.stop()
    tx.wait()
    
    print("\nTransmission complete!")
    
    # Analyze if requested
    if args.analyze:
        print("\nAnalyzing transmitted signal...")
        analyze_modulation(
            '/tmp/am_transmit.iq',
            args.sample_rate,
            min(args.duration, 1.0)
        )
    
    # Print file info
    import os
    if os.path.exists('/tmp/am_transmit.iq'):
        file_size = os.path.getsize('/tmp/am_transmit.iq')
        print(f"\nOutput file size: {file_size/1e6:.1f} MB")
        print(f"Samples written: {file_size//8:,}")


if __name__ == '__main__':
    main()