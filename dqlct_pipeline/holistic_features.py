# holistic_features.py
"""
Standard Hilbert Transform Quaternion Representation for Speech.
Converts audio to quaternion signal using analytic signal decomposition.
Uses: q[n] = Re{fa[n]} + Im{fa[n]}i + 0j + 0k
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from quaternion_core import Quaternion, create_quaternion_array


class HilbertQuaternionFeatures:
    """
    Extracts quaternion representation using standard Hilbert transform.

    Standard Hilbert approach:
    A(t) = f(t) + i·H{f(t)}
    q[n] = Re{fa[n]} + Im{fa[n]}i + 0j + 0k
    """

    def __init__(self, sr=16000, frame_length=512, hop_length=256):
        """
        Initialize feature extractor.

        Args:
            sr: Sample rate in Hz
            frame_length: Frame size for analysis (samples)
            hop_length: Hop size between frames (samples)
        """
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length

        print(f"Initialized HilbertQuaternionFeatures (Standard Method):")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Frame length: {frame_length} samples ({frame_length / sr * 1000:.1f} ms)")
        print(f"  Hop length: {hop_length} samples ({hop_length / sr * 1000:.1f} ms)")
        print(f"  Overlap: {(1 - hop_length / frame_length) * 100:.1f}%")
        print(f"  Method: q[n] = Re{{fa[n]}} + Im{{fa[n]}}i + 0j + 0k")

    def audio_to_quaternion_signal(self, audio, verbose=True):
        """
        Convert raw audio to quaternion signal using standard Hilbert transform.

        Standard method:
        1. Apply Hilbert to audio → analytic signal
        2. q[n] = Re{fa[n]} + Im{fa[n]}i + 0j + 0k

        Args:
            audio: 1D numpy array of audio samples
            verbose: Print progress messages

        Returns:
            quat_signal: List of Quaternion objects, one per sample
        """
        if verbose:
            print("\nExtracting standard Hilbert-based quaternion features...")

        # Step 1: Hilbert transform of signal
        if verbose:
            print("  - Computing Hilbert transform of audio...")
        analytic_signal = hilbert(audio)
        w_component = np.real(analytic_signal)  # Real part: original signal
        x_component = np.imag(analytic_signal)  # Imaginary part: Hilbert transform
        y_component = np.zeros_like(audio)      # j component = 0
        z_component = np.zeros_like(audio)      # k component = 0

        # Normalize components to similar scale
        if verbose:
            print("  - Normalizing components...")

        def normalize_component(comp):
            max_val = np.max(np.abs(comp))
            if max_val > 0:
                return comp / max_val
            return comp

        w_component = normalize_component(w_component)
        x_component = normalize_component(x_component)

        # Create quaternion signal
        if verbose:
            print("  - Creating quaternion signal...")

        quat_signal = []
        for i in range(len(audio)):
            q = Quaternion(
                w=float(w_component[i]),
                x=float(x_component[i]),
                y=0.0,  # j = 0
                z=0.0   # k = 0
            )
            quat_signal.append(q)

        if verbose:
            print(f"  ✓ Created {len(quat_signal)} quaternion samples")
            print(f"    Duration: {len(quat_signal) / self.sr:.2f} seconds")
            print(f"    Format: q = w + xi + 0j + 0k (standard Hilbert)")

        return quat_signal

    def visualize_features(self, audio, quat_signal, save_path=None):
        """
        Visualize the quaternion components from standard Hilbert transform.
        Creates a 3-panel plot (original, w, x components).

        Args:
            audio: Original audio signal
            quat_signal: List of Quaternion objects
            save_path: Optional path to save figure
        """
        # Extract components
        w_vals = [q.w for q in quat_signal]
        x_vals = [q.x for q in quat_signal]

        # Time axis
        time_axis = np.arange(len(quat_signal)) / self.sr

        # Create 3-panel plot
        fig, axes = plt.subplots(3, 1, figsize=(14, 9))

        # Panel 1: Original waveform
        axes[0].plot(time_axis, audio, linewidth=0.5, color='black')
        axes[0].set_title('Original Audio Waveform', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)

        # Panel 2: W-component (Real part)
        axes[1].plot(time_axis, w_vals, color='blue', linewidth=1.5)
        axes[1].fill_between(time_axis, w_vals, alpha=0.3, color='blue')
        axes[1].set_title('W-component: Real Part (Original Signal)',
                          fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)

        # Panel 3: X-component (Imaginary part - Hilbert)
        axes[2].plot(time_axis, x_vals, color='green', linewidth=1.5)
        axes[2].fill_between(time_axis, x_vals, alpha=0.3, color='green')
        axes[2].set_title('X-component: Imaginary Part (Hilbert Transform)',
                          fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved figure to {save_path}")

        plt.show()

        # Print statistics
        print("\n" + "=" * 60)
        print("QUATERNION COMPONENT STATISTICS (STANDARD HILBERT)")
        print("=" * 60)
        print(f"W (Real):    mean={np.mean(w_vals):.4f}, std={np.std(w_vals):.4f}, "
              f"min={np.min(w_vals):.4f}, max={np.max(w_vals):.4f}")
        print(f"X (Hilbert): mean={np.mean(x_vals):.4f}, std={np.std(x_vals):.4f}, "
              f"min={np.min(x_vals):.4f}, max={np.max(x_vals):.4f}")
        print(f"Y (j):       All zeros (standard method)")
        print(f"Z (k):       All zeros (standard method)")
        print("=" * 60)


# Test function
if __name__ == "__main__":
    print("=" * 70)
    print("STANDARD HILBERT TRANSFORM QUATERNION EXTRACTION TEST")
    print("=" * 70)

    # Generate test audio or load file
    audio_file = "test_2_wav.wav"

    try:
        audio, sr = librosa.load(audio_file, sr=16000)
        print(f"\n✓ Loaded: {audio_file}")
        print(f"  Duration: {len(audio) / sr:.2f} seconds")
        print(f"  Sample rate: {sr} Hz")
    except Exception as e:
        print(f"\n⚠ Could not load {audio_file}: {e}")
        print("  Generating synthetic speech signal...")

        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Simulate speech with formants and modulation
        f0, f1, f2 = 150, 500, 1500
        audio = (np.sin(2 * np.pi * f0 * t) *
                 (np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)))

        # Add amplitude envelope
        envelope = np.exp(-2 * np.abs(t - duration / 2))
        audio = audio * envelope * 0.3

    # Initialize extractor
    extractor = HilbertQuaternionFeatures(
        sr=sr,
        frame_length=512,
        hop_length=256
    )

    # Extract quaternion signal
    quat_signal = extractor.audio_to_quaternion_signal(audio)

    # Show first few samples
    print("\nFirst 5 quaternion samples:")
    for i in range(min(5, len(quat_signal))):
        print(f"  Sample {i}: {quat_signal[i]}")

    # Visualize
    extractor.visualize_features(audio, quat_signal)

    print("\n" + "=" * 70)
    print("STANDARD HILBERT TRANSFORM QUATERNION EXTRACTION TEST COMPLETE")
    print("=" * 70)