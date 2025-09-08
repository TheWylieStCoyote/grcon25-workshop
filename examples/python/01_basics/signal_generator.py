#!/usr/bin/env python3
"""
Multi-Waveform Signal Generator
Demonstrates various signal types and parameter control in GNU Radio
Usage: python3 signal_generator.py --waveform sine --frequency 1000 --amplitude 0.5
"""

import argparse
import sys
import signal as sig
import time
from gnuradio import gr
from gnuradio import blocks
from gnuradio import analog
from gnuradio import audio
import numpy as np

class SignalGenerator(gr.top_block):
    def __init__(self, waveform='sine', frequency=1000, amplitude=0.5, 
                 sample_rate=48000, duration=None, output_device=''):
        gr.top_block.__init__(self, "Multi-Waveform Signal Generator")
        
        # Parameters
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.amplitude = amplitude
        self.waveform = waveform
        self.duration = duration
        
        # Create signal source based on waveform type
        if waveform == 'sine':
            self.source = analog.sig_source_f(
                sample_rate, analog.GR_COS_WAVE, frequency, amplitude, 0, 0)
        elif waveform == 'square':
            self.source = analog.sig_source_f(
                sample_rate, analog.GR_SQR_WAVE, frequency, amplitude, 0, 0)
        elif waveform == 'triangle':
            self.source = analog.sig_source_f(
                sample_rate, analog.GR_TRI_WAVE, frequency, amplitude, 0, 0)
        elif waveform == 'sawtooth':
            self.source = analog.sig_source_f(
                sample_rate, analog.GR_SAW_WAVE, frequency, amplitude, 0, 0)
        elif waveform == 'noise':
            self.source = blocks.noise_source_f(
                blocks.GR_GAUSSIAN, amplitude, 0)
        elif waveform == 'impulse':
            # Create impulse train
            self.source = blocks.vector_source_f(
                [1.0] + [0.0] * int(sample_rate/frequency - 1), True)
            self.scale = blocks.multiply_const_ff(amplitude)
        elif waveform == 'chirp':
            # Create frequency sweep (chirp)
            self.source = analog.sig_source_f(
                sample_rate, analog.GR_COS_WAVE, frequency, amplitude, 0, 0)
            self.sweep_rate = 100  # Hz per second
        else:
            raise ValueError(f"Unknown waveform type: {waveform}")
        
        # Audio sink
        self.audio_sink = audio.sink(sample_rate, output_device, True)
        
        # Optional: Add file sink for recording
        self.file_sink = None
        
        # Connections
        if waveform == 'impulse':
            self.connect(self.source, self.scale, self.audio_sink)
        else:
            self.connect(self.source, self.audio_sink)
        
        # For chirp, we'll update frequency in a thread
        if waveform == 'chirp':
            self.chirp_thread_running = True
            import threading
            self.chirp_thread = threading.Thread(target=self.chirp_update)
            self.chirp_thread.daemon = True
    
    def chirp_update(self):
        """Update frequency for chirp signal"""
        start_freq = self.frequency
        end_freq = self.frequency * 2
        sweep_time = 2.0  # seconds
        
        while self.chirp_thread_running:
            for t in np.linspace(0, sweep_time, int(sweep_time * 50)):
                if not self.chirp_thread_running:
                    break
                freq = start_freq + (end_freq - start_freq) * t / sweep_time
                self.source.set_frequency(freq)
                time.sleep(0.02)
            
            # Reverse sweep
            for t in np.linspace(0, sweep_time, int(sweep_time * 50)):
                if not self.chirp_thread_running:
                    break
                freq = end_freq - (end_freq - start_freq) * t / sweep_time
                self.source.set_frequency(freq)
                time.sleep(0.02)
    
    def start(self):
        """Start the flowgraph and chirp thread if needed"""
        super().start()
        if self.waveform == 'chirp' and hasattr(self, 'chirp_thread'):
            self.chirp_thread.start()
    
    def stop(self):
        """Stop the flowgraph and chirp thread if needed"""
        if self.waveform == 'chirp':
            self.chirp_thread_running = False
            if hasattr(self, 'chirp_thread'):
                self.chirp_thread.join(timeout=1)
        super().stop()
    
    def set_frequency(self, frequency):
        """Update signal frequency"""
        self.frequency = frequency
        if hasattr(self.source, 'set_frequency'):
            self.source.set_frequency(frequency)
    
    def set_amplitude(self, amplitude):
        """Update signal amplitude"""
        self.amplitude = amplitude
        if hasattr(self.source, 'set_amplitude'):
            self.source.set_amplitude(amplitude)
        elif hasattr(self, 'scale'):
            self.scale.set_k(amplitude)
    
    def add_file_sink(self, filename):
        """Add file sink for recording"""
        self.disconnect_all()
        self.file_sink = blocks.file_sink(gr.sizeof_float, filename, False)
        self.file_sink.set_unbuffered(False)
        
        if self.waveform == 'impulse':
            self.connect(self.source, self.scale)
            self.connect(self.scale, self.audio_sink)
            self.connect(self.scale, self.file_sink)
        else:
            self.connect(self.source, self.audio_sink)
            self.connect(self.source, self.file_sink)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='GNU Radio Multi-Waveform Signal Generator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-w', '--waveform', 
                       choices=['sine', 'square', 'triangle', 'sawtooth', 
                               'noise', 'impulse', 'chirp'],
                       default='sine',
                       help='Waveform type to generate')
    
    parser.add_argument('-f', '--frequency', 
                       type=float, 
                       default=1000,
                       help='Signal frequency in Hz')
    
    parser.add_argument('-a', '--amplitude', 
                       type=float, 
                       default=0.3,
                       help='Signal amplitude (0.0 to 1.0)')
    
    parser.add_argument('-s', '--sample-rate', 
                       type=int, 
                       default=48000,
                       help='Sample rate in Hz')
    
    parser.add_argument('-d', '--duration', 
                       type=float, 
                       default=None,
                       help='Duration in seconds (None for continuous)')
    
    parser.add_argument('-o', '--output', 
                       type=str, 
                       default=None,
                       help='Output filename for recording')
    
    parser.add_argument('--device', 
                       type=str, 
                       default='',
                       help='Audio output device')
    
    parser.add_argument('-i', '--interactive', 
                       action='store_true',
                       help='Interactive mode with parameter control')
    
    return parser.parse_args()

def interactive_mode(tb):
    """Run interactive parameter control"""
    print("\nInteractive Mode Commands:")
    print("  f <freq>  - Set frequency (Hz)")
    print("  a <amp>   - Set amplitude (0-1)")
    print("  q         - Quit")
    print()
    
    while True:
        try:
            cmd = input("Command: ").strip().lower()
            if cmd.startswith('f '):
                freq = float(cmd.split()[1])
                tb.set_frequency(freq)
                print(f"Frequency set to {freq} Hz")
            elif cmd.startswith('a '):
                amp = float(cmd.split()[1])
                amp = max(0, min(1, amp))  # Clamp to [0, 1]
                tb.set_amplitude(amp)
                print(f"Amplitude set to {amp}")
            elif cmd == 'q':
                break
            else:
                print("Unknown command")
        except (ValueError, IndexError):
            print("Invalid input")
        except KeyboardInterrupt:
            break

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Validate parameters
    if not 0 <= args.amplitude <= 1:
        print("Error: Amplitude must be between 0 and 1")
        sys.exit(1)
    
    if args.frequency <= 0:
        print("Error: Frequency must be positive")
        sys.exit(1)
    
    # Create signal generator
    tb = SignalGenerator(
        waveform=args.waveform,
        frequency=args.frequency,
        amplitude=args.amplitude,
        sample_rate=args.sample_rate,
        duration=args.duration,
        output_device=args.device
    )
    
    # Add file sink if output specified
    if args.output:
        tb.add_file_sink(args.output)
        print(f"Recording to: {args.output}")
    
    # Signal handler for clean shutdown
    def signal_handler(signum, frame):
        print("\nShutting down...")
        tb.stop()
        tb.wait()
        sys.exit(0)
    
    sig.signal(sig.SIGINT, signal_handler)
    
    # Start the flowgraph
    tb.start()
    
    # Print info
    print("GNU Radio Signal Generator")
    print("-" * 40)
    print(f"Waveform:    {args.waveform}")
    print(f"Frequency:   {args.frequency} Hz")
    print(f"Amplitude:   {args.amplitude}")
    print(f"Sample Rate: {args.sample_rate} Hz")
    print(f"Duration:    {args.duration or 'Continuous'} seconds")
    print("-" * 40)
    
    if args.waveform == 'chirp':
        print("Generating frequency sweep (chirp)...")
    
    print("Press Ctrl+C to stop")
    print()
    
    try:
        if args.interactive:
            interactive_mode(tb)
        elif args.duration:
            time.sleep(args.duration)
        else:
            tb.wait()
    except KeyboardInterrupt:
        pass
    
    # Clean shutdown
    tb.stop()
    tb.wait()
    
    if args.output:
        print(f"\nSignal saved to: {args.output}")

if __name__ == '__main__':
    main()