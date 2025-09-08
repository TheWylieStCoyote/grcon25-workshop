#!/usr/bin/env python3
"""
File Operations with IQ Data
Demonstrates reading and writing complex IQ data files in GNU Radio
Usage: python3 file_operations.py --mode [record|playback|convert]
"""

import argparse
import sys
import os
import struct
import numpy as np
from gnuradio import gr
from gnuradio import blocks
from gnuradio import analog
from gnuradio import audio
from gnuradio.filter import firdes

class IQFileRecorder(gr.top_block):
    """Record IQ data from various sources"""
    
    def __init__(self, filename, sample_rate=1e6, frequency=100e6, 
                 duration=10, source_type='simulation'):
        gr.top_block.__init__(self, "IQ File Recorder")
        
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.duration = duration
        
        # Create source based on type
        if source_type == 'simulation':
            # Simulated signal: carrier + modulation
            self.carrier = analog.sig_source_c(
                sample_rate, analog.GR_COS_WAVE, 100e3, 0.5, 0, 0)
            self.modulation = analog.sig_source_f(
                sample_rate, analog.GR_COS_WAVE, 1e3, 0.3, 0, 0)
            self.multiplier = blocks.multiply_cc()
            self.float_to_complex = blocks.float_to_complex()
            
            # Create AM modulated signal
            self.connect(self.modulation, (self.float_to_complex, 0))
            self.connect(blocks.null_source(gr.sizeof_float), 
                        (self.float_to_complex, 1))
            self.connect(self.carrier, (self.multiplier, 0))
            self.connect(self.float_to_complex, (self.multiplier, 1))
            
            self.source = self.multiplier
        else:
            # For hardware sources, would add RTL-SDR, USRP, etc.
            raise NotImplementedError(f"Source type {source_type} not implemented")
        
        # File sink with metadata
        self.file_sink = blocks.file_sink(
            gr.sizeof_gr_complex, filename, False)
        self.file_sink.set_unbuffered(False)
        
        # Throttle for simulation
        if source_type == 'simulation':
            self.throttle = blocks.throttle(gr.sizeof_gr_complex, sample_rate)
            self.connect(self.source, self.throttle, self.file_sink)
        else:
            self.connect(self.source, self.file_sink)
        
        # Save metadata
        self.save_metadata(filename)
    
    def save_metadata(self, filename):
        """Save metadata file with recording parameters"""
        metadata_file = filename.replace('.iq', '.meta')
        if not metadata_file.endswith('.meta'):
            metadata_file += '.meta'
        
        with open(metadata_file, 'w') as f:
            f.write(f"# GNU Radio IQ Recording Metadata\n")
            f.write(f"sample_rate={self.sample_rate}\n")
            f.write(f"center_frequency={self.frequency}\n")
            f.write(f"duration={self.duration}\n")
            f.write(f"data_type=complex64\n")
            f.write(f"byte_order=little_endian\n")

class IQFilePlayer(gr.top_block):
    """Playback IQ data files"""
    
    def __init__(self, filename, sample_rate=1e6, repeat=False, 
                 output_type='audio'):
        gr.top_block.__init__(self, "IQ File Player")
        
        self.sample_rate = sample_rate
        
        # Check if file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"IQ file not found: {filename}")
        
        # Load metadata if available
        metadata = self.load_metadata(filename)
        if metadata:
            self.sample_rate = metadata.get('sample_rate', sample_rate)
        
        # File source
        self.file_source = blocks.file_source(
            gr.sizeof_gr_complex, filename, repeat)
        
        # Throttle to control playback rate
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, self.sample_rate)
        
        if output_type == 'audio':
            # Demodulate and play as audio
            # Simple AM demodulation
            self.complex_to_mag = blocks.complex_to_mag()
            
            # Resample to audio rate
            audio_rate = 48000
            decimation = int(self.sample_rate / audio_rate)
            if decimation > 1:
                taps = firdes.low_pass(1, self.sample_rate, 
                                      audio_rate/2, audio_rate/10)
                self.resampler = blocks.fir_filter_fff(decimation, taps)
                self.audio_sink = audio.sink(audio_rate, '', True)
                
                self.connect(self.file_source, self.throttle)
                self.connect(self.throttle, self.complex_to_mag)
                self.connect(self.complex_to_mag, self.resampler)
                self.connect(self.resampler, self.audio_sink)
            else:
                self.audio_sink = audio.sink(int(self.sample_rate), '', True)
                self.connect(self.file_source, self.throttle)
                self.connect(self.throttle, self.complex_to_mag)
                self.connect(self.complex_to_mag, self.audio_sink)
        
        elif output_type == 'null':
            # Just read the file (for testing)
            self.null_sink = blocks.null_sink(gr.sizeof_gr_complex)
            self.connect(self.file_source, self.throttle, self.null_sink)
    
    def load_metadata(self, filename):
        """Load metadata file if it exists"""
        metadata_file = filename.replace('.iq', '.meta')
        if not metadata_file.endswith('.meta'):
            metadata_file += '.meta'
        
        if os.path.exists(metadata_file):
            metadata = {}
            with open(metadata_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=')
                        try:
                            metadata[key] = float(value)
                        except ValueError:
                            metadata[key] = value
            return metadata
        return None

class IQFileConverter:
    """Convert between different IQ file formats"""
    
    @staticmethod
    def convert_to_wav(input_file, output_file, sample_rate=1e6):
        """Convert IQ file to WAV format"""
        import wave
        
        # Read IQ data
        data = np.fromfile(input_file, dtype=np.complex64)
        
        # Demodulate (simple AM)
        audio_data = np.abs(data)
        
        # Normalize
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Write WAV file
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(audio_data.tobytes())
        
        print(f"Converted {input_file} to {output_file}")
    
    @staticmethod
    def convert_to_csv(input_file, output_file, max_samples=10000):
        """Convert IQ file to CSV format (for analysis)"""
        # Read IQ data
        data = np.fromfile(input_file, dtype=np.complex64)
        
        # Limit samples for CSV
        if len(data) > max_samples:
            data = data[:max_samples]
            print(f"Limited to first {max_samples} samples")
        
        # Write CSV
        with open(output_file, 'w') as f:
            f.write("Index,I,Q,Magnitude,Phase\n")
            for i, sample in enumerate(data):
                mag = np.abs(sample)
                phase = np.angle(sample)
                f.write(f"{i},{sample.real},{sample.imag},{mag},{phase}\n")
        
        print(f"Converted {input_file} to {output_file}")
    
    @staticmethod
    def analyze_file(filename):
        """Analyze IQ file and print statistics"""
        # Read IQ data
        data = np.fromfile(filename, dtype=np.complex64)
        
        if len(data) == 0:
            print("File is empty")
            return
        
        # Calculate statistics
        magnitude = np.abs(data)
        phase = np.angle(data)
        
        print(f"\nIQ File Analysis: {filename}")
        print("-" * 50)
        print(f"Number of samples: {len(data):,}")
        print(f"File size: {os.path.getsize(filename):,} bytes")
        print(f"Data type: complex64 (8 bytes per sample)")
        print("\nMagnitude Statistics:")
        print(f"  Min:    {np.min(magnitude):.6f}")
        print(f"  Max:    {np.max(magnitude):.6f}")
        print(f"  Mean:   {np.mean(magnitude):.6f}")
        print(f"  StdDev: {np.std(magnitude):.6f}")
        print("\nPhase Statistics (radians):")
        print(f"  Min:    {np.min(phase):.6f}")
        print(f"  Max:    {np.max(phase):.6f}")
        print(f"  Mean:   {np.mean(phase):.6f}")
        print("\nI/Q Statistics:")
        print(f"  I range: [{np.min(data.real):.6f}, {np.max(data.real):.6f}]")
        print(f"  Q range: [{np.min(data.imag):.6f}, {np.max(data.imag):.6f}]")
        
        # Detect DC offset
        dc_offset = np.mean(data)
        if np.abs(dc_offset) > 0.01:
            print(f"\nWarning: DC offset detected: {dc_offset}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='GNU Radio IQ File Operations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--mode', 
                       choices=['record', 'playback', 'convert', 'analyze'],
                       default='analyze',
                       help='Operation mode')
    
    parser.add_argument('-i', '--input', 
                       type=str,
                       help='Input filename')
    
    parser.add_argument('-o', '--output', 
                       type=str,
                       help='Output filename')
    
    parser.add_argument('-s', '--sample-rate', 
                       type=float,
                       default=1e6,
                       help='Sample rate in Hz')
    
    parser.add_argument('-f', '--frequency', 
                       type=float,
                       default=100e6,
                       help='Center frequency in Hz (for recording)')
    
    parser.add_argument('-d', '--duration', 
                       type=float,
                       default=10,
                       help='Recording duration in seconds')
    
    parser.add_argument('--format', 
                       choices=['wav', 'csv'],
                       default='wav',
                       help='Output format for conversion')
    
    parser.add_argument('--repeat', 
                       action='store_true',
                       help='Repeat playback')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if args.mode == 'record':
        if not args.output:
            print("Error: Output filename required for recording")
            sys.exit(1)
        
        # Ensure .iq extension
        if not args.output.endswith('.iq'):
            args.output += '.iq'
        
        print(f"Recording IQ data to: {args.output}")
        print(f"Sample rate: {args.sample_rate/1e6:.1f} MHz")
        print(f"Duration: {args.duration} seconds")
        
        tb = IQFileRecorder(
            args.output,
            sample_rate=args.sample_rate,
            frequency=args.frequency,
            duration=args.duration
        )
        
        tb.start()
        import time
        time.sleep(args.duration)
        tb.stop()
        tb.wait()
        
        file_size = os.path.getsize(args.output)
        print(f"\nRecording complete!")
        print(f"File size: {file_size/1e6:.1f} MB")
        print(f"Samples: {file_size//8:,}")
    
    elif args.mode == 'playback':
        if not args.input:
            print("Error: Input filename required for playback")
            sys.exit(1)
        
        print(f"Playing IQ file: {args.input}")
        
        tb = IQFilePlayer(
            args.input,
            sample_rate=args.sample_rate,
            repeat=args.repeat
        )
        
        tb.start()
        
        try:
            tb.wait()
        except KeyboardInterrupt:
            print("\nStopping playback...")
        
        tb.stop()
        tb.wait()
    
    elif args.mode == 'convert':
        if not args.input or not args.output:
            print("Error: Input and output filenames required for conversion")
            sys.exit(1)
        
        converter = IQFileConverter()
        
        if args.format == 'wav':
            converter.convert_to_wav(args.input, args.output, args.sample_rate)
        elif args.format == 'csv':
            converter.convert_to_csv(args.input, args.output)
    
    elif args.mode == 'analyze':
        if not args.input:
            print("Error: Input filename required for analysis")
            sys.exit(1)
        
        IQFileConverter.analyze_file(args.input)

if __name__ == '__main__':
    main()