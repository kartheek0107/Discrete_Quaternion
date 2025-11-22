# dqlct_transform.py
"""
Discrete Quaternion Linear Canonical Transform (DQLCT)
Direct implementation with proper quaternion kernel multiplication.
Based on the mathematically correct formulation from the research documents.
"""

import numpy as np
from quaternion_core import Quaternion, create_quaternion_array


class QLCT1D:
    """
    1D Discrete Quaternion Linear Canonical Transform

    Implements the transform with ABCD matrix parameters:
    X(u) = (1/√N) * Σ K(n,m) * x(n)

    where K(n,m) = exp_j(φ(n,m)) is the quaternion kernel
    """

    def __init__(self, N, a, b, c, d, dt=1.0):
        """
        Initialize DQLCT with matrix parameters.

        Args:
            N: Signal length
            a, b, c, d: ABCD matrix parameters (must satisfy ad - bc = 1)
            dt: Time sampling interval
        """
        self.N = int(N)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.dt = float(dt)

        # Verify unimodular condition
        det_A = self.a * self.d - self.b * self.c
        if abs(det_A - 1.0) > 1e-10:
            raise ValueError(
                f"Matrix must be unimodular (det = 1), got det = {det_A:.6f}\n"
                f"Matrix: [[{a}, {b}], [{c}, {d}]]"
            )

        # Check b != 0 requirement
        if abs(self.b) < 1e-14:
            raise NotImplementedError(
                "This implementation requires b ≠ 0.\n"
                "For b = 0, use special case formulation."
            )

        # Frequency domain sampling interval
        self.du = 2.0 * np.pi * self.b / (self.N * self.dt)

        print(f"Initialized QLCT1D:")
        print(f"  N = {self.N}")
        print(f"  Matrix A = [[{self.a:.4f}, {self.b:.4f}],")
        print(f"              [{self.c:.4f}, {self.d:.4f}]]")
        print(f"  det(A) = {det_A:.6f}")
        print(f"  dt = {self.dt:.6f}")
        print(f"  du = {self.du:.6f}")

    def _compute_chirps(self):
        """
        Compute input and output chirp multipliers.
        These are used in the fast algorithm (not implemented yet).

        Returns:
            input_chirps, output_chirps: Arrays of Quaternion objects
        """
        input_chirps = np.empty(self.N, dtype=object)
        output_chirps = np.empty(self.N, dtype=object)

        for n in range(self.N):
            phase_in = (self.a * (n ** 2) * (self.dt ** 2)) / (2.0 * self.b)
            input_chirps[n] = Quaternion.exp_j(phase_in)

        for m in range(self.N):
            phase_out = (self.d * (m ** 2) * (self.du ** 2)) / (2.0 * self.b)
            output_chirps[m] = Quaternion.exp_j(phase_out)

        return input_chirps, output_chirps

    def direct_transform(self, signal):
        """
        Direct DQLCT forward transform (O(N²) complexity).

        X(m) = (1/√N) * Σ_{n=0}^{N-1} x(n) * K(n,m)

        where K(n,m) = exp_j(φ(n,m)) with:
        φ(n,m) = (a*n²*dt²)/(2b) - (2π*m*n)/N + (d*m²*du²)/(2b)

        Args:
            signal: Array of Quaternion objects or convertible data

        Returns:
            spectrum: Array of Quaternion objects
        """
        # Ensure signal is quaternion array
        if not isinstance(signal, np.ndarray) or signal.dtype != object:
            signal = create_quaternion_array(signal)

        if len(signal) != self.N:
            raise ValueError(
                f"Signal length {len(signal)} must match N={self.N}"
            )

        result = np.empty(self.N, dtype=object)
        norm_factor = 1.0 / np.sqrt(self.N)

        # Compute transform
        for m in range(self.N):
            acc = Quaternion(0.0, 0.0, 0.0, 0.0)

            for n in range(self.N):
                # Compute phase
                phase_total = (
                        (self.a * n ** 2 * self.dt ** 2) / (2.0 * self.b) -
                        (2.0 * np.pi * m * n) / self.N +
                        (self.d * m ** 2 * self.du ** 2) / (2.0 * self.b)
                )

                # Create kernel (exp_j rotates in j-axis)
                kernel = Quaternion.exp_j(phase_total)

                # RIGHT multiplication: signal * kernel
                term = signal[n] * kernel
                acc = acc + term

            result[m] = norm_factor * acc

        return result

    def inverse_transform(self, spectrum):
        """
        Direct DQLCT inverse transform (O(N²) complexity).

        x(n) = (1/√N) * Σ_{m=0}^{N-1} X(m) * K*(n,m)

        where K*(n,m) is the conjugated kernel.

        Args:
            spectrum: Array of Quaternion objects

        Returns:
            signal: Reconstructed array of Quaternion objects
        """
        # Ensure spectrum is quaternion array
        if not isinstance(spectrum, np.ndarray) or spectrum.dtype != object:
            spectrum = create_quaternion_array(spectrum)

        if len(spectrum) != self.N:
            raise ValueError(
                f"Spectrum length {len(spectrum)} must match N={self.N}"
            )

        result = np.empty(self.N, dtype=object)
        norm_factor = 1.0 / np.sqrt(self.N)

        # Compute inverse transform
        for n in range(self.N):
            acc = Quaternion(0, 0, 0, 0)

            for m in range(self.N):
                # Same phase as forward transform
                phase_total = (
                        (self.a * n ** 2 * self.dt ** 2) / (2.0 * self.b) -
                        (2.0 * np.pi * m * n) / self.N +
                        (self.d * m ** 2 * self.du ** 2) / (2.0 * self.b)
                )

                # Conjugated kernel for inverse
                kernel_conj = Quaternion.exp_j(-phase_total)

                # RIGHT multiplication: spectrum * kernel_conj
                term = spectrum[m] * kernel_conj
                acc = acc + term

            result[n] = norm_factor * acc

        return result


def create_standard_matrices():
    """
    Create standard DQLCT matrix configurations.

    Returns:
        dict: Dictionary of (name, [a, b, c, d]) pairs
    """
    matrices = {
        'QFT': [0.0, 1.0, -1.0, 0.0],
        'Fractional_45deg': [
            np.cos(np.pi / 4), np.sin(np.pi / 4),
            -np.sin(np.pi / 4), np.cos(np.pi / 4)
        ],
        'Fractional_30deg': [
            np.cos(np.pi / 6), np.sin(np.pi / 6),
            -np.sin(np.pi / 6), np.cos(np.pi / 6)
        ],
        'Custom': [1.0, 0.5, 0.0, 1.0],
        'Identity': [1.0, 0.0, 0.0, 1.0],  # Note: b=0, will raise error
    }
    return matrices


# Test and validation functions
def test_energy_conservation(qlct, signal):
    """
    Test energy conservation (Parseval's theorem).

    For proper unitary transform:
    Σ|x(n)|² = Σ|X(m)|²
    """
    spectrum = qlct.direct_transform(signal)

    # Compute energies
    input_energy = sum(q.norm() ** 2 for q in signal)
    output_energy = sum(q.norm() ** 2 for q in spectrum)

    # Relative error
    error = abs(input_energy - output_energy) / (input_energy + 1e-20)

    return input_energy, output_energy, error


def test_reconstruction(qlct, signal):
    """
    Test perfect reconstruction via inverse transform.

    x(n) = IDQLCT(DQLCT(x(n)))
    """
    # Forward transform
    spectrum = qlct.direct_transform(signal)

    # Inverse transform
    reconstructed = qlct.inverse_transform(spectrum)

    # Compute reconstruction error
    errors = [(signal[i] - reconstructed[i]).norm() for i in range(len(signal))]
    mean_error = np.mean(errors)
    max_error = np.max(errors)

    return reconstructed, mean_error, max_error


def test_linearity(qlct, signal1, signal2, alpha=2.0, beta=3.0):
    """
    Test linearity property.

    DQLCT(α*x₁ + β*x₂) = α*DQLCT(x₁) + β*DQLCT(x₂)
    """
    # Compute linear combination of signals
    combined = np.array([
        alpha * signal1[i] + beta * signal2[i]
        for i in range(len(signal1))
    ], dtype=object)

    # Transform combined signal
    F_combined = qlct.direct_transform(combined)

    # Transform individual signals and combine
    F1 = qlct.direct_transform(signal1)
    F2 = qlct.direct_transform(signal2)
    F_linear = np.array([
        alpha * F1[i] + beta * F2[i]
        for i in range(len(F1))
    ], dtype=object)

    # Compute error
    errors = [(F_combined[i] - F_linear[i]).norm() for i in range(len(F_combined))]
    mean_error = np.mean(errors)

    return mean_error


def validate_dqlct(qlct, test_signal):
    """
    Run complete validation suite on DQLCT transform.

    Args:
        qlct: QLCT1D instance
        test_signal: Test signal (array of Quaternions)

    Returns:
        dict: Validation results
    """
    print("\n" + "=" * 60)
    print("DQLCT VALIDATION SUITE")
    print("=" * 60)

    results = {}

    # Test 1: Energy conservation
    print("\n1. Energy Conservation Test (Parseval's Theorem)")
    E_in, E_out, energy_error = test_energy_conservation(qlct, test_signal)
    results['energy_in'] = E_in
    results['energy_out'] = E_out
    results['energy_error'] = energy_error

    print(f"   Input energy:  {E_in:.6f}")
    print(f"   Output energy: {E_out:.6f}")
    print(f"   Relative error: {energy_error:.6e}")

    if energy_error < 1e-12:
        print("   ✓ PASS: Perfect energy conservation")
    elif energy_error < 1e-6:
        print("   ⚠ WARNING: Small energy error (numerical precision)")
    else:
        print("   ✗ FAIL: Significant energy error")

    # Test 2: Reconstruction
    print("\n2. Reconstruction Test (Inverse Transform)")
    reconstructed, mean_error, max_error = test_reconstruction(qlct, test_signal)
    results['reconstruction_mean_error'] = mean_error
    results['reconstruction_max_error'] = max_error

    print(f"   Mean reconstruction error: {mean_error:.6e}")
    print(f"   Max reconstruction error:  {max_error:.6e}")

    if max_error < 1e-12:
        print("   ✓ PASS: Perfect reconstruction")
    elif max_error < 1e-6:
        print("   ⚠ WARNING: Small reconstruction error")
    else:
        print("   ✗ FAIL: Significant reconstruction error")

    # Test 3: Linearity
    print("\n3. Linearity Test")
    # Create second test signal
    signal2 = create_quaternion_array([
        Quaternion(1, 1, 0, 0) for _ in range(len(test_signal))
    ])

    linearity_error = test_linearity(qlct, test_signal, signal2)
    results['linearity_error'] = linearity_error

    print(f"   Linearity error: {linearity_error:.6e}")

    if linearity_error < 1e-12:
        print("   ✓ PASS: Linear transform")
    elif linearity_error < 1e-6:
        print("   ⚠ WARNING: Small linearity error")
    else:
        print("   ✗ FAIL: Non-linear behavior detected")

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum([
        energy_error < 1e-6,
        max_error < 1e-6,
        linearity_error < 1e-6
    ])

    print(f"Tests passed: {passed}/3")

    if passed == 3:
        print("✓ All tests PASSED - DQLCT implementation is correct")
    elif passed >= 2:
        print("⚠ Most tests passed - Minor numerical issues")
    else:
        print("✗ FAILED - Implementation has errors")

    print("=" * 60)

    return results


# Main test
if __name__ == "__main__":
    print("=" * 70)
    print("DQLCT TRANSFORM MODULE TEST")
    print("=" * 70)

    # Test parameters
    N = 64

    # Create test signal (quaternion impulse)
    test_signal = [
        Quaternion(1, 0.5, 0.3, 0.2) if i == 0 else Quaternion(0, 0, 0, 0)
        for i in range(N)
    ]
    test_signal = create_quaternion_array(test_signal)

    print(f"\nTest signal: {N} samples")
    print(f"First sample: {test_signal[0]}")

    # Test with QFT matrix
    print("\n" + "=" * 60)
    print("Testing with QFT matrix")
    print("=" * 60)

    qlct = QLCT1D(N, a=0.0, b=1.0, c=-1.0, d=0.0)

    # Run validation
    results = validate_dqlct(qlct, test_signal)

    # Show spectrum
    spectrum = qlct.direct_transform(test_signal)
    print("\nFirst 5 spectrum values:")
    for i in range(5):
        print(f"  X[{i}] = {spectrum[i]}")

    print("\n" + "=" * 70)
    print("DQLCT TRANSFORM MODULE TEST COMPLETE")
    print("=" * 70)