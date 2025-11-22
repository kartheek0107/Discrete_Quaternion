# quaternion_core.py
"""
Core quaternion classes and operations for DQLCT speech processing.
Provides basic quaternion arithmetic and array operations.
"""

import numpy as np


class Quaternion:
    """
    Quaternion class with full arithmetic operations.
    Represents q = w + xi + yj + zk
    """
    __slots__ = ('w', 'x', 'y', 'z')

    def __init__(self, w=0.0, x=0.0, y=0.0, z=0.0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other):
        """Quaternion addition"""
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w + other.w,
                self.x + other.x,
                self.y + other.y,
                self.z + other.z
            )
        else:
            return Quaternion(self.w + other, self.x, self.y, self.z)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Quaternion subtraction"""
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w - other.w,
                self.x - other.x,
                self.y - other.y,
                self.z - other.z
            )
        else:
            return Quaternion(self.w - other, self.x, self.y, self.z)

    def __mul__(self, other):
        """Quaternion multiplication (Hamilton product)"""
        if isinstance(other, (int, float)):
            return Quaternion(
                self.w * other,
                self.x * other,
                self.y * other,
                self.z * other
            )
        elif isinstance(other, Quaternion):
            # Hamilton product: q1 * q2
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)
        else:
            raise TypeError(f"Cannot multiply Quaternion with {type(other)}")

    def __rmul__(self, other):
        """Right multiplication by scalar"""
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        else:
            raise TypeError(f"Cannot multiply {type(other)} with Quaternion")

    def conjugate(self):
        """Quaternion conjugate: q* = w - xi - yj - zk"""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self):
        """Quaternion norm: |q| = sqrt(w² + x² + y² + z²)"""
        return np.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self):
        """Return normalized quaternion"""
        n = self.norm()
        if n == 0:
            return Quaternion(0, 0, 0, 0)
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)

    @staticmethod
    def exp_j(theta):
        """
        Exponential in j-axis: exp(j*theta) = cos(theta) + j*sin(theta)
        Used in DQLCT kernel
        """
        return Quaternion(np.cos(theta), 0.0, np.sin(theta), 0.0)

    @staticmethod
    def exp_i(theta):
        """
        Exponential in i-axis: exp(i*theta) = cos(theta) + i*sin(theta)
        """
        return Quaternion(np.cos(theta), np.sin(theta), 0.0, 0.0)

    def __repr__(self):
        return f"Q({self.w:.4f} + {self.x:.4f}i + {self.y:.4f}j + {self.z:.4f}k)"

    def __abs__(self):
        return self.norm()

    def to_array(self):
        """Convert to numpy array [w, x, y, z]"""
        return np.array([self.w, self.x, self.y, self.z])


def create_quaternion_array(data):
    """
    Convert various input formats to quaternion array.

    Args:
        data: Can be:
            - List of Quaternion objects
            - List of 4-tuples/lists
            - numpy array of shape (..., 4)
            - numpy array of floats (converted to real quaternions)

    Returns:
        numpy array of Quaternion objects
    """
    if data is None:
        raise TypeError("Input is None")

    # Already quaternion list
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], Quaternion):
        return np.array(data, dtype=object)

    arr = np.asarray(data)

    # Array with shape (..., 4) - interpret last dimension as quaternion components
    if arr.ndim >= 1 and arr.dtype != object and arr.shape[-1] == 4:
        flat_shape = arr.shape[:-1]
        result = np.empty(flat_shape, dtype=object)
        for idx in np.ndindex(flat_shape):
            vals = arr[idx]
            result[idx] = Quaternion(
                float(vals[0]), float(vals[1]),
                float(vals[2]), float(vals[3])
            )
        return result.reshape(flat_shape)

    # Regular array - convert to real quaternions (w component only)
    else:
        flat = arr.ravel()
        result = np.empty(flat.shape, dtype=object)
        for i, val in enumerate(flat):
            result[i] = Quaternion(float(val), 0.0, 0.0, 0.0)
        return result.reshape(arr.shape)


def quaternion_array_to_components(quat_array):
    """
    Extract component arrays from quaternion array.

    Args:
        quat_array: numpy array of Quaternion objects

    Returns:
        w, x, y, z: Four numpy arrays
    """
    w = np.array([q.w for q in quat_array.flat]).reshape(quat_array.shape)
    x = np.array([q.x for q in quat_array.flat]).reshape(quat_array.shape)
    y = np.array([q.y for q in quat_array.flat]).reshape(quat_array.shape)
    z = np.array([q.z for q in quat_array.flat]).reshape(quat_array.shape)
    return w, x, y, z


def components_to_quaternion_array(w, x, y, z):
    """
    Create quaternion array from component arrays.

    Args:
        w, x, y, z: numpy arrays of same shape

    Returns:
        numpy array of Quaternion objects
    """
    assert w.shape == x.shape == y.shape == z.shape, "Component shapes must match"

    result = np.empty(w.shape, dtype=object)
    for idx in np.ndindex(w.shape):
        result[idx] = Quaternion(
            float(w[idx]), float(x[idx]),
            float(y[idx]), float(z[idx])
        )
    return result


# Test functions
if __name__ == "__main__":
    print("=" * 60)
    print("QUATERNION CORE MODULE TEST")
    print("=" * 60)

    # Test basic operations
    q1 = Quaternion(1, 2, 3, 4)
    q2 = Quaternion(5, 6, 7, 8)

    print(f"\nq1 = {q1}")
    print(f"q2 = {q2}")
    print(f"q1 + q2 = {q1 + q2}")
    print(f"q1 * q2 = {q1 * q2}")
    print(f"|q1| = {q1.norm():.4f}")
    print(f"q1* = {q1.conjugate()}")

    # Test exp functions
    theta = np.pi / 4
    print(f"\nexp_j(π/4) = {Quaternion.exp_j(theta)}")
    print(f"exp_i(π/4) = {Quaternion.exp_i(theta)}")

    # Test array conversion
    print("\n" + "=" * 60)
    print("Array conversion test:")
    data = [[1, 2, 3, 4], [5, 6, 7, 8]]
    quat_arr = create_quaternion_array(data)
    print(f"Input: {data}")
    print(f"Quaternion array: {quat_arr}")

    # Test component extraction
    w, x, y, z = quaternion_array_to_components(quat_arr)
    print(f"\nExtracted components:")
    print(f"w = {w}")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"z = {z}")

    print("\n" + "=" * 60)
    print("QUATERNION CORE MODULE TEST COMPLETE")
    print("=" * 60)