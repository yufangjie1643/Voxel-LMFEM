"""Generate auditable scalar HEX8 arithmetic and matched Numba CPU sweeps.

Usage: python generate_kernels.py
No symbolic algebra or runtime matrix construction is used in apply kernels.
"""
from pathlib import Path
from math import sqrt


OUT = Path(__file__).with_name("generated_kernels.py")
PAIRS = [(i, j) for i in range(6) for j in range(i, 6)]
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def ci(i, j):
    return PAIRS.index((min(i, j), max(i, j)))


def add(lines, expression):
    lines.append("    " + expression)


def strain_terms(mode):
    result = []
    for i, j in VOIGT:
        terms = []
        for component, direction in ([(i, i)] if i == j else [(i, j), (j, i)]):
            if not mode & (1 << direction):
                terms.append((mode | (1 << direction), component, direction))
        result.append(terms)
    return result


def butterfly(lines, source, prefix, transpose=False):
    values = [[source(a, d) for d in range(3)] for a in range(8)]
    for stage, stride in enumerate((1, 2, 4)):
        updated = [[None] * 3 for _ in range(8)]
        for a in range(8):
            if a & stride:
                continue
            b = a | stride
            for d in range(3):
                first, second = values[a][d], values[b][d]
                names = [f"{prefix}{stage}_{a}_{d}", f"{prefix}{stage}_{b}_{d}"]
                expressions = ((f"{first} - {second}", f"{first} + {second}")
                               if transpose else
                               (f"{first} + {second}", f"{second} - {first}"))
                for name, expression in zip(names, expressions):
                    add(lines, f"{name} = {expression}")
                updated[a][d], updated[b][d] = names
        values = updated
    return values


def modal_local(kind):
    lines = ["@njit(cache=True, inline='always', fastmath=False)",
             f"def {kind}_modal_local(u, c, scale, g):"]
    q = butterfly(lines, lambda a, d: f"u[{3*a+d}]", "q")
    force = [[[] for _ in range(3)] for _ in range(8)]
    for mode in ([0] if kind == "hg" else [0, 1, 2, 4, 3, 5, 6]):
        terms = strain_terms(mode)
        active = [i for i, row in enumerate(terms) if row]
        for i in active:
            add(lines, f"e{mode}_{i} = " + " + ".join(
                f"g[{d}] * {q[m][component]}" for m, component, d in terms[i]))
        weight = 3 ** mode.bit_count()
        for i in active:
            add(lines, f"s{mode}_{i} = (" + " + ".join(
                f"c[{ci(i,j)}] * e{mode}_{j}" for j in active) + f") * (g[3] / {float(weight)})")
            for m, component, d in terms[i]:
                force[m][component].append(f"g[{d}] * s{mode}_{i}")
    for m in range(8):
        for d in range(3):
            if kind == "hg" and m in (3, 5, 6, 7):
                force[m][d].append(f"(scale * 0.125) * {q[m][d]}")
            add(lines, f"p{m}_{d} = " + (" + ".join(force[m][d]) or "0.0"))
    f = butterfly(lines, lambda a, d: f"p{a}_{d}", "f", transpose=True)
    add(lines, "return (" + ", ".join(f[a][d] for a in range(8) for d in range(3)) + ")")
    return "\n".join(lines)


def reference_local(kind):
    lines = ["@njit(cache=True, inline='always', fastmath=False)",
             f"def {kind}_reference_local(u, c, scale, g):"]
    forces = [[] for _ in range(24)]
    points = [(0.0, 0.0, 0.0)] if kind == "hg" else [
        tuple((1.0 if n & (1 << d) else -1.0) / sqrt(3.0) for d in range(3))
        for n in range(8)]
    for point_index, point in enumerate(points):
        derivatives = []
        for a in range(8):
            signs = [1 if a & (1 << d) else -1 for d in range(3)]
            values = []
            for d in range(3):
                multiplier = float(signs[d])
                for j in range(3):
                    if j != d:
                        multiplier *= 1.0 + signs[j] * point[j]
                values.append(f"({multiplier!r} * g[{d}])")
            derivatives.append(values)
        strain_rows = [[] for _ in range(6)]
        for i, (first, second) in enumerate(VOIGT):
            for a in range(8):
                strain_rows[i].append(f"{derivatives[a][second]} * u[{3*a+first}]")
                if first != second:
                    strain_rows[i].append(f"{derivatives[a][first]} * u[{3*a+second}]")
        for i, terms in enumerate(strain_rows):
            add(lines, f"e{point_index}_{i} = " + " + ".join(terms))
        for i in range(6):
            add(lines, f"s{point_index}_{i} = (" + " + ".join(
                f"c[{ci(i,j)}] * e{point_index}_{j}" for j in range(6)) +
                f") * (g[3] / {float(len(points))})")
        for a in range(8):
            for i, (first, second) in enumerate(VOIGT):
                forces[3*a+first].append(f"{derivatives[a][second]} * s{point_index}_{i}")
                if first != second:
                    forces[3*a+second].append(f"{derivatives[a][first]} * s{point_index}_{i}")
    if kind == "hg":
        for m in (3, 5, 6, 7):
            for d in range(3):
                terms = [f"{(-1 if ((m & (~a & 7)).bit_count() % 2) else 1)}.0 * u[{3*a+d}]"
                         for a in range(8)]
                add(lines, f"h{m}_{d} = (" + " + ".join(terms) + ") * (scale * 0.125)")
                for a in range(8):
                    sign = -1 if ((m & (~a & 7)).bit_count() % 2) else 1
                    forces[3*a+d].append(f"{sign}.0 * h{m}_{d}")
    add(lines, "return (" + ",\n        ".join(" + ".join(terms) for terms in forces) + ")")
    return "\n".join(lines)


SWEEP = '''
@njit(cache=True, parallel=True, fastmath=False)
def NAME_element_pass(u, c, scales, shape, g, forces):
    nz, ny, nx = shape
    for element_index in prange(nz * ny * nx):
        element = np.int64(element_index)
        z, remainder = divmod(element, ny * nx)
        y, x = divmod(remainder, nx)
        local_u = np.empty(24, dtype=np.float64)
        for a in range(8):
            xx = (x + (a & 1)) % nx
            yy = (y + ((a >> 1) & 1)) % ny
            zz = (z + ((a >> 2) & 1)) % nz
            node = (zz * ny + yy) * nx + xx
            for d in range(3):
                local_u[3*a+d] = u[node, d]
        local_f = NAME_local(local_u, c[element], scales[element], g)
        for j in range(24):
            forces[element, j] = local_f[j]


@njit(cache=True, parallel=True, fastmath=False)
def NAME_tile_pass(u, c, scales, shape, g, output, tile):
    nz, ny, nx = shape
    tz, ty, tx = (nz + tile - 1)//tile, (ny + tile - 1)//tile, (nx + tile - 1)//tile
    for tile_index in prange(tz * ty * tx):
        tile_id = np.int64(tile_index)
        bz, rem = divmod(tile_id, ty * tx)
        by, bx = divmod(rem, tx)
        sz, sy, sx = bz * tile, by * tile, bx * tile
        dz, dy, dx = min(tile, nz-sz), min(tile, ny-sy), min(tile, nx-sx)
        # Private per CPU worker; no N_voxel x 24 global intermediate.
        local_forces = np.empty((dz+1, dy+1, dx+1, 24), dtype=np.float64)
        local_u = np.empty(24, dtype=np.float64)
        for ez in range(dz+1):
            for ey in range(dy+1):
                for ex in range(dx+1):
                    z, y, x = (sz+ez-1) % nz, (sy+ey-1) % ny, (sx+ex-1) % nx
                    element = (z * ny + y) * nx + x
                    for a in range(8):
                        xx = (x + (a & 1)) % nx
                        yy = (y + ((a >> 1) & 1)) % ny
                        zz = (z + ((a >> 2) & 1)) % nz
                        node = (zz * ny + yy) * nx + xx
                        for d in range(3):
                            local_u[3*a+d] = u[node, d]
                    local_f = NAME_local(local_u, c[element], scales[element], g)
                    for j in range(24):
                        local_forces[ez, ey, ex, j] = local_f[j]
        for iz in range(dz):
            for iy in range(dy):
                for ix in range(dx):
                    node = ((sz+iz) * ny + sy+iy) * nx + sx+ix
                    for d in range(3):
                        value = 0.0
                        for a in range(8):
                            ez, ey, ex = iz+1-((a>>2)&1), iy+1-((a>>1)&1), ix+1-(a&1)
                            value += local_forces[ez, ey, ex, 3*a+d]
                        output[node, d] = value
'''


def main():
    parts = ['"""Generated by generate_kernels.py; FP64, fastmath=False, lexicographic x-fast corners."""',
             "import numpy as np\nfrom numba import njit, prange"]
    for kind in ("hg", "fi"):
        parts.extend([reference_local(kind), modal_local(kind)])
        for method in ("reference", "modal"):
            parts.append(SWEEP.replace("NAME", f"{kind}_{method}"))
    OUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"Generated {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
