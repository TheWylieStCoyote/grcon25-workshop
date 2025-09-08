#!/usr/bin/env python3
"""
Interpolator Block Example - 1:N Input/Output Ratio
Demonstrates creating custom interpolation blocks in GNU Radio
Usage: python3 interpolator_block.py
"""

import numpy as np
from gnuradio import gr
from gnuradio import blocks
from gnuradio import analog
from gnuradio.filter import firdes
import matplotlib.pyplot as plt
import time

class ZeroStuffingInterpolator(gr.interp_block):
    """
    Simple zero-stuffing interpolator
    Inserts zeros between samples and applies filtering
    """
    
    def __init__(self, interpolation_factor=4):
        """
        Initialize zero-stuffing interpolator
        
        Args:
            interpolation_factor: Number of output samples per input sample
        """
        gr.interp_block.__init__(
            self,
            name="Zero Stuffing Interpolator",
            in_sig=[np.float32],
            out_sig=[np.float32],
            interp=interpolation_factor
        )
        
        self.interpolation = interpolation_factor
        
        # Design interpolation filter (low-pass)
        self.filter_taps = firdes.low_pass(
            interpolation_factor,            # Gain
            interpolation_factor,            # Sample rate (normalized)
            0.4,                            # Cutoff frequency
            0.1,                            # Transition width
            firdes.WIN_HAMMING)
        
        # Filter state
        self.filter_state = np.zeros(len(self.filter_taps) - 1)
        
    def work(self, input_items, output_items):
        """Insert zeros and filter"""
        in0 = input_items[0]
        out0 = output_items[0]
        
        # Zero-stuff the input
        zero_stuffed = np.zeros(len(in0) * self.interpolation)
        zero_stuffed[::self.interpolation] = in0
        
        # Apply interpolation filter
        # Concatenate with filter state for continuity
        full_signal = np.concatenate([self.filter_state, zero_stuffed])
        filtered = np.convolve(full_signal, self.filter_taps, mode='valid')
        
        # Update filter state
        if len(zero_stuffed) >= len(self.filter_state):
            self.filter_state = zero_stuffed[-len(self.filter_state):]
        else:
            self.filter_state = np.concatenate([
                self.filter_state[len(zero_stuffed):], zero_stuffed])
        
        # Copy to output
        out0[:len(filtered)] = filtered[:len(out0)]
        
        return len(in0)


class LinearInterpolator(gr.interp_block):
    """
    Linear interpolation between samples
    Creates smooth transitions between input samples
    """
    
    def __init__(self, interpolation_factor=4):
        """
        Initialize linear interpolator
        
        Args:
            interpolation_factor: Number of output samples per input sample
        """
        gr.interp_block.__init__(
            self,
            name="Linear Interpolator",
            in_sig=[np.float32],
            out_sig=[np.float32],
            interp=interpolation_factor
        )
        
        self.interpolation = interpolation_factor
        self.previous_sample = 0
        
    def work(self, input_items, output_items):
        """Linearly interpolate between samples"""
        in0 = input_items[0]
        out0 = output_items[0]
        
        out_idx = 0
        for i in range(len(in0)):
            # Current and next sample
            current = in0[i]
            
            if i == 0:
                previous = self.previous_sample
            else:
                previous = in0[i-1]
            
            # Linear interpolation
            for j in range(self.interpolation):
                alpha = j / self.interpolation
                out0[out_idx] = previous * (1 - alpha) + current * alpha
                out_idx += 1
        
        # Save last sample for next call
        if len(in0) > 0:
            self.previous_sample = in0[-1]
        
        return len(in0)


class SplineInterpolator(gr.interp_block):
    """
    Cubic spline interpolation for smooth curves
    Provides smoother output than linear interpolation
    """
    
    def __init__(self, interpolation_factor=4):
        """
        Initialize spline interpolator
        
        Args:
            interpolation_factor: Number of output samples per input sample
        """
        gr.interp_block.__init__(
            self,
            name="Spline Interpolator",
            in_sig=[np.float32],
            out_sig=[np.float32],
            interp=interpolation_factor
        )
        
        self.interpolation = interpolation_factor
        self.buffer = np.zeros(4)  # Need 4 points for cubic spline
        
    def cubic_interpolate(self, y0, y1, y2, y3, mu):
        """
        Cubic interpolation between y1 and y2
        y0 and y3 are used for calculating slopes
        mu is the fractional position between y1 and y2
        """
        mu2 = mu * mu
        a0 = y3 - y2 - y0 + y1
        a1 = y0 - y1 - a0
        a2 = y2 - y0
        a3 = y1
        
        return a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3
    
    def work(self, input_items, output_items):
        """Apply cubic spline interpolation"""
        in0 = input_items[0]
        out0 = output_items[0]
        
        # Need at least 2 samples to interpolate
        if len(in0) < 2:
            out0[:] = 0
            return 0
        
        out_idx = 0
        for i in range(len(in0)):
            # Update buffer
            self.buffer = np.roll(self.buffer, -1)
            self.buffer[-1] = in0[i]
            
            # Only interpolate after we have enough samples
            if i >= 1:
                # Get the 4 points for interpolation
                if i >= 3:
                    y0, y1, y2, y3 = self.buffer
                elif i == 2:
                    y0 = self.buffer[1]
                    y1, y2, y3 = self.buffer[1:]
                else:  # i == 1
                    y0 = self.buffer[2]
                    y1 = self.buffer[2]
                    y2, y3 = self.buffer[2:]
                
                # Interpolate between y1 and y2
                for j in range(self.interpolation):
                    mu = j / self.interpolation
                    out0[out_idx] = self.cubic_interpolate(y0, y1, y2, y3, mu)
                    out_idx += 1
        
        return len(in0)


class SincInterpolator(gr.interp_block):
    """
    Sinc interpolation (theoretically perfect for band-limited signals)
    Uses windowed sinc function for practical implementation
    """
    
    def __init__(self, interpolation_factor=4, num_taps=32):
        """
        Initialize sinc interpolator
        
        Args:
            interpolation_factor: Upsampling ratio
            num_taps: Number of sinc function taps per side
        """
        gr.interp_block.__init__(
            self,
            name="Sinc Interpolator",
            in_sig=[np.float32],
            out_sig=[np.float32],
            interp=interpolation_factor
        )
        
        self.interpolation = interpolation_factor
        self.num_taps = num_taps
        
        # Generate windowed sinc kernel
        self.generate_sinc_kernel()
        
        # Buffer for input samples
        self.sample_buffer = np.zeros(2 * num_taps)
        
    def generate_sinc_kernel(self):
        """Generate windowed sinc interpolation kernel"""
        # Create oversampled sinc function
        oversample = 16  # Oversample factor for kernel
        kernel_len = 2 * self.num_taps * oversample
        
        t = np.linspace(-self.num_taps, self.num_taps, kernel_len)
        
        # Sinc function with Blackman window
        with np.errstate(divide='ignore', invalid='ignore'):
            sinc = np.sin(np.pi * t) / (np.pi * t)
            sinc[np.isnan(sinc)] = 1.0  # Handle t=0
        
        # Apply window
        window = np.blackman(kernel_len)
        self.sinc_kernel = sinc * window
        self.kernel_step = oversample
        
    def sinc_interp_sample(self, samples, frac_delay):
        """
        Interpolate single sample using sinc function
        frac_delay is fractional sample delay (0 to 1)
        """
        # Find kernel index for this fractional delay
        kernel_idx = int(frac_delay * self.kernel_step)
        
        # Get kernel samples around this point
        center = len(self.sinc_kernel) // 2
        start = center - self.num_taps * self.kernel_step + kernel_idx
        
        # Extract kernel values at integer positions
        kernel_vals = self.sinc_kernel[start::self.kernel_step][:len(samples)]
        
        # Compute interpolated value
        return np.dot(samples, kernel_vals)
    
    def work(self, input_items, output_items):
        """Apply sinc interpolation"""
        in0 = input_items[0]
        out0 = output_items[0]
        
        out_idx = 0
        for i in range(len(in0)):
            # Update buffer
            self.sample_buffer = np.roll(self.sample_buffer, -1)
            self.sample_buffer[-1] = in0[i]
            
            # Interpolate
            for j in range(self.interpolation):
                frac = j / self.interpolation
                out0[out_idx] = self.sinc_interp_sample(
                    self.sample_buffer, frac)
                out_idx += 1
        
        return len(in0)


class RepeatInterpolator(gr.interp_block):
    """
    Simple repeat/hold interpolator
    Repeats each input sample N times (zero-order hold)
    """
    
    def __init__(self, interpolation_factor=4):
        """
        Initialize repeat interpolator
        
        Args:
            interpolation_factor: Number of times to repeat each sample
        """
        gr.interp_block.__init__(
            self,
            name="Repeat Interpolator",
            in_sig=[np.float32],
            out_sig=[np.float32],
            interp=interpolation_factor
        )
        
        self.interpolation = interpolation_factor
        
    def work(self, input_items, output_items):
        """Repeat each input sample"""
        in0 = input_items[0]
        out0 = output_items[0]
        
        out_idx = 0
        for sample in in0:
            for _ in range(self.interpolation):
                out0[out_idx] = sample
                out_idx += 1
        
        return len(in0)


def create_test_flowgraph():
    """Create test flowgraph for interpolator blocks"""
    
    class InterpolatorTestFlowgraph(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self, "Interpolator Test")
            
            # Parameters
            input_rate = 1000  # Low rate for visible interpolation
            interpolation = 8
            
            # Create low-rate test signal
            self.sig_source = analog.sig_source_f(
                input_rate, analog.GR_COS_WAVE, 50, 1.0, 0)
            
            # Add some noise for realism
            self.noise = blocks.noise_source_f(blocks.GR_GAUSSIAN, 0.05, 0)
            self.adder = blocks.add_vff(1)
            
            # Throttle at input rate
            self.throttle = blocks.throttle(gr.sizeof_float, input_rate)
            
            # Different interpolators
            self.zero_stuff = ZeroStuffingInterpolator(interpolation)
            self.linear = LinearInterpolator(interpolation)
            self.spline = SplineInterpolator(interpolation)
            self.sinc = SincInterpolator(interpolation, 16)
            self.repeat = RepeatInterpolator(interpolation)
            
            # File sinks
            self.original_sink = blocks.file_sink(
                gr.sizeof_float, '/tmp/interp_original.dat', False)
            self.zero_sink = blocks.file_sink(
                gr.sizeof_float, '/tmp/interp_zero.dat', False)
            self.linear_sink = blocks.file_sink(
                gr.sizeof_float, '/tmp/interp_linear.dat', False)
            self.spline_sink = blocks.file_sink(
                gr.sizeof_float, '/tmp/interp_spline.dat', False)
            self.sinc_sink = blocks.file_sink(
                gr.sizeof_float, '/tmp/interp_sinc.dat', False)
            self.repeat_sink = blocks.file_sink(
                gr.sizeof_float, '/tmp/interp_repeat.dat', False)
            
            # Connections
            self.connect(self.sig_source, (self.adder, 0))
            self.connect(self.noise, (self.adder, 1))
            self.connect(self.adder, self.throttle)
            
            # Branch to different interpolators
            self.connect(self.throttle, self.original_sink)
            self.connect(self.throttle, self.zero_stuff, self.zero_sink)
            self.connect(self.throttle, self.linear, self.linear_sink)
            self.connect(self.throttle, self.spline, self.spline_sink)
            self.connect(self.throttle, self.sinc, self.sinc_sink)
            self.connect(self.throttle, self.repeat, self.repeat_sink)
    
    return InterpolatorTestFlowgraph()


def analyze_interpolation_results(input_rate=1000, interpolation=8):
    """Analyze and plot interpolation results"""
    
    # Read data files
    try:
        original = np.fromfile('/tmp/interp_original.dat', dtype=np.float32)
        zero_stuff = np.fromfile('/tmp/interp_zero.dat', dtype=np.float32)
        linear = np.fromfile('/tmp/interp_linear.dat', dtype=np.float32)
        spline = np.fromfile('/tmp/interp_spline.dat', dtype=np.float32)
        sinc = np.fromfile('/tmp/interp_sinc.dat', dtype=np.float32)
        repeat = np.fromfile('/tmp/interp_repeat.dat', dtype=np.float32)
    except FileNotFoundError:
        print("Data files not found. Run the interpolator test first.")
        return
    
    # Limit samples for plotting
    plot_samples = min(50, len(original))
    interp_samples = plot_samples * interpolation
    
    # Create time axes
    t_original = np.arange(plot_samples) / input_rate
    t_interp = np.arange(interp_samples) / (input_rate * interpolation)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Interpolation Methods Comparison', fontsize=14)
    
    # Original signal
    axes[0, 0].stem(t_original, original[:plot_samples], basefmt=' ', 
                   linefmt='b-', markerfmt='bo')
    axes[0, 0].set_title('Original Signal (Low Rate)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Repeat (Zero-order hold)
    axes[0, 1].plot(t_interp, repeat[:interp_samples], 'r-', linewidth=1)
    axes[0, 1].stem(t_original, original[:plot_samples], basefmt=' ', 
                   linefmt='b--', markerfmt='bo', alpha=0.5)
    axes[0, 1].set_title('Repeat/Hold Interpolation')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Linear interpolation
    axes[1, 0].plot(t_interp, linear[:interp_samples], 'g-', linewidth=1)
    axes[1, 0].stem(t_original, original[:plot_samples], basefmt=' ', 
                   linefmt='b--', markerfmt='bo', alpha=0.5)
    axes[1, 0].set_title('Linear Interpolation')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Spline interpolation
    axes[1, 1].plot(t_interp, spline[:interp_samples], 'm-', linewidth=1)
    axes[1, 1].stem(t_original, original[:plot_samples], basefmt=' ', 
                   linefmt='b--', markerfmt='bo', alpha=0.5)
    axes[1, 1].set_title('Cubic Spline Interpolation')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Sinc interpolation
    axes[2, 0].plot(t_interp, sinc[:interp_samples], 'c-', linewidth=1)
    axes[2, 0].stem(t_original, original[:plot_samples], basefmt=' ', 
                   linefmt='b--', markerfmt='bo', alpha=0.5)
    axes[2, 0].set_title('Sinc Interpolation')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Amplitude')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Zero-stuffing with filtering
    axes[2, 1].plot(t_interp, zero_stuff[:interp_samples], 'orange', linewidth=1)
    axes[2, 1].stem(t_original, original[:plot_samples], basefmt=' ', 
                   linefmt='b--', markerfmt='bo', alpha=0.5)
    axes[2, 1].set_title('Zero-Stuffing + LPF')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Amplitude')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Second figure for frequency domain comparison
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8))
    fig2.suptitle('Frequency Domain Analysis', fontsize=14)
    
    # Compute FFTs
    from scipy import signal
    
    methods = [
        ('Original', original, input_rate),
        ('Repeat', repeat, input_rate * interpolation),
        ('Linear', linear, input_rate * interpolation),
        ('Spline', spline, input_rate * interpolation),
        ('Sinc', sinc, input_rate * interpolation),
        ('Zero-Stuff', zero_stuff, input_rate * interpolation)
    ]
    
    for idx, (name, data, rate) in enumerate(methods):
        ax = axes2[idx // 3, idx % 3]
        
        # Compute spectrum
        f, psd = signal.welch(data, rate, nperseg=min(256, len(data)//4))
        
        ax.semilogy(f, psd)
        ax.set_title(f'{name} Spectrum')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, rate/2])
    
    plt.tight_layout()
    plt.show()
    
    # Print quality metrics
    print("\nInterpolation Quality Metrics:")
    print("-" * 50)
    
    # Calculate RMS error compared to ideal (sinc) interpolation
    min_len = min(len(sinc), len(repeat), len(linear), len(spline), len(zero_stuff))
    
    if min_len > 0:
        sinc_ref = sinc[:min_len]
        
        rms_repeat = np.sqrt(np.mean((repeat[:min_len] - sinc_ref)**2))
        rms_linear = np.sqrt(np.mean((linear[:min_len] - sinc_ref)**2))
        rms_spline = np.sqrt(np.mean((spline[:min_len] - sinc_ref)**2))
        rms_zero = np.sqrt(np.mean((zero_stuff[:min_len] - sinc_ref)**2))
        
        print(f"RMS Error (vs Sinc interpolation):")
        print(f"  Repeat/Hold:  {rms_repeat:.6f}")
        print(f"  Linear:       {rms_linear:.6f}")
        print(f"  Cubic Spline: {rms_spline:.6f}")
        print(f"  Zero-Stuff:   {rms_zero:.6f}")


def main():
    """Main function to demonstrate interpolator blocks"""
    
    print("GNU Radio Custom Interpolator Block Examples")
    print("=" * 50)
    print("\nThis example demonstrates various interpolation methods:")
    print("1. Repeat/Hold - Simple sample repetition")
    print("2. Linear - Linear interpolation between samples")
    print("3. Cubic Spline - Smooth spline interpolation")
    print("4. Sinc - Theoretically perfect (band-limited)")
    print("5. Zero-Stuffing - Insert zeros and filter")
    print("\nTest signal: 50 Hz cosine with noise")
    print(f"Input rate: 1000 Hz")
    print(f"Interpolation factor: 8")
    print(f"Output rate: 8000 Hz")
    print("=" * 50)
    
    # Create and run flowgraph
    tb = create_test_flowgraph()
    
    print("\nRunning interpolation test for 2 seconds...")
    tb.start()
    time.sleep(2)
    tb.stop()
    tb.wait()
    
    print("\nInterpolation complete!")
    print("\nAnalyzing results...")
    
    # Analyze and plot results
    analyze_interpolation_results()
    
    print("\nDone!")


if __name__ == '__main__':
    main()