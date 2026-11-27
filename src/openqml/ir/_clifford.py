"""Conjugate independent Hermitian Pauli words in parallel, one bit per word."""

import numpy as np

CLIFFORD_GATES = frozenset("id x y z h s sdg sx sxdg cx cy cz swap iswap".split())


def packed_rows(columns, count):
    """Transpose little-endian bit columns in strips, preserving wire order."""
    if not columns:
        yield from (0 for _ in range(count))
        return
    stride = (count + 7) // 8
    packed = np.frombuffer(b"".join(col.to_bytes(stride, "little") for col in columns), np.uint8)
    packed = packed.reshape(len(columns), stride)
    for start in range(0, count, 256):
        size = min(256, count - start)
        strip = packed[:, start // 8 : (start + size + 7) // 8]
        bits = np.unpackbits(strip, axis=1, count=size, bitorder="little")
        rows = np.packbits(bits, axis=0, bitorder="little").T.copy()
        for row in rows:
            yield int.from_bytes(row, "little")


class PackedPaulis:
    """Each column holds X and Z bits on a wire. Each sign bit negates one word."""

    def __init__(self):
        self.columns = {}
        self.sign = 0

    def apply(self, name, wires, *, inverse=False):
        if name == "id":
            return
        a = wires[0]
        x, z = self.columns.get(a, (0, 0))
        if len(wires) == 1:
            if not (x or z):
                return
            if name == "h":
                self.sign ^= x & z
                self.columns[a] = z, x
            elif name in {"s", "sdg"}:
                self.sign ^= x & (~z if (name == "sdg") ^ inverse else z)
                self.columns[a] = x, z ^ x
            elif name in {"sx", "sxdg"}:
                # H S H (or its inverse), with the Hermitian Y sign included.
                self.sign ^= z & (x if (name == "sxdg") ^ inverse else ~x)
                self.columns[a] = x ^ z, z
            elif name == "x":
                self.sign ^= z
            elif name == "z":
                self.sign ^= x
            elif name == "y":
                self.sign ^= x ^ z
            else:
                raise ValueError(f"Unknown one-wire Clifford gate {name!r}.")
            return
        b = wires[1]
        u, v = self.columns.get(b, (0, 0))
        if not (x or z or u or v):
            return
        if name == "cx":
            self.sign ^= x & v & ~(u ^ z)
            self.columns[a], self.columns[b] = (x, z ^ v), (u ^ x, v)
        elif name == "swap":
            self.columns[a], self.columns[b] = (u, v), (x, z)
        elif name == "cy":
            self.apply("sdg", (b,))
            self.apply("cx", wires)
            self.apply("s", (b,))
        elif name == "cz":
            self.apply("h", (b,))
            self.apply("cx", wires)
            self.apply("h", (b,))
        elif name == "iswap":
            # SWAP, S x S and CZ commute.
            self.apply("swap", wires)
            self.apply("s", (a,), inverse=inverse)
            self.apply("s", (b,), inverse=inverse)
            self.apply("cz", wires)
        else:
            raise ValueError(f"Unknown two-wire Clifford gate {name!r}.")

    def propagate(self, operations, *, inverse=False):
        for op in reversed(operations) if inverse else operations:
            if op.name not in {"barrier", "measure"}:
                self.apply(op.name, op.qubits, inverse=inverse)

    def rows(self, wires, count):
        columns = [self.columns.get(q, (0, 0)) for q in wires]
        xs = packed_rows([x for x, _ in columns], count)
        zs = packed_rows([z for _, z in columns], count)
        for i, (x, z) in enumerate(zip(xs, zs, strict=True)):
            # Convert (-1)^sign i^#Y X^x Z^z to i^phase X^x Z^z.
            yield x, z, (2 * ((self.sign >> i) & 1) + (x & z).bit_count()) % 4
