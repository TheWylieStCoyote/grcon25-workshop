#!/usr/bin/env python3
"""
Simple GNU Radio Flowgraph in Python
Demonstrates basic signal generation and visualization
"""

from gnuradio import gr
from gnuradio import blocks
from gnuradio import analog
from gnuradio import audio
from gnuradio.filter import firdes
import sys
import signal
import time

class SimpleFlowgraph(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self, "Simple Flowgraph")
        
        # Variables
        self.samp_rate = samp_rate = 32000
        self.frequency = frequency = 440  # A4 note
        self.amplitude = amplitude = 0.3
        
        # Blocks
        self.audio_sink = audio.sink(samp_rate, '', True)
        self.analog_sig_source = analog.sig_source_f(
            samp_rate,
            analog.GR_COS_WAVE,
            frequency,
            amplitude,
            0,
            0
        )
        
        # Optional: Add noise
        self.noise_source = blocks.noise_source_f(
            blocks.GR_GAUSSIAN,
            0.01,  # Low noise amplitude
            0
        )
        
        self.adder = blocks.add_vff(1)
        
        # Connections
        self.connect((self.analog_sig_source, 0), (self.adder, 0))
        self.connect((self.noise_source, 0), (self.adder, 1))
        self.connect((self.adder, 0), (self.audio_sink, 0))
    
    def set_frequency(self, frequency):
        self.frequency = frequency
        self.analog_sig_source.set_frequency(self.frequency)
    
    def set_amplitude(self, amplitude):
        self.amplitude = amplitude
        self.analog_sig_source.set_amplitude(self.amplitude)
    
    def get_frequency(self):
        return self.frequency
    
    def get_amplitude(self):
        return self.amplitude

def main():
    # Create flowgraph
    tb = SimpleFlowgraph()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nShutting down...")
        tb.stop()
        tb.wait()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start flowgraph
    tb.start()
    print("GNU Radio Simple Flowgraph")
    print("-" * 30)
    print(f"Sample Rate: {tb.samp_rate} Hz")
    print(f"Frequency: {tb.frequency} Hz")
    print(f"Amplitude: {tb.amplitude}")
    print("-" * 30)
    print("Playing 440 Hz tone (A4 note)")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        # Play different notes
        notes = {
            'A4': 440,
            'C5': 523,
            'E5': 659,
            'G5': 784,
        }
        
        for note, freq in notes.items():
            print(f"Playing {note} ({freq} Hz)...")
            tb.set_frequency(freq)
            time.sleep(1)
        
        # Return to A4 and keep playing
        print("Returning to A4...")
        tb.set_frequency(440)
        
        # Wait forever (until Ctrl+C)
        tb.wait()
        
    except KeyboardInterrupt:
        pass
    
    # Stop flowgraph
    tb.stop()
    tb.wait()

if __name__ == '__main__':
    main()