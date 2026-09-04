"""Generate the butane fixture geometry in both PDB and GRO form."""

import numpy as np

from ffrefine.io.structure import Structure, write_multimodel_pdb

NAMES = [
    "C1",
    "H11",
    "H12",
    "H13",
    "C2",
    "H21",
    "H22",
    "C3",
    "H31",
    "H32",
    "C4",
    "H41",
    "H42",
    "H43",
]
ELEMENTS = ["C", "H", "H", "H", "C", "H", "H", "C", "H", "H", "C", "H", "H", "H"]

# All-trans n-butane, Angstrom, generated from ideal internal coordinates.
COORDS_A = np.array(
    [
        [-1.9264, 0.2646, 0.0000],
        [-2.0324, 0.8974, -0.8842],
        [-2.0324, 0.8974, 0.8842],
        [-2.7440, -0.4570, 0.0000],
        [-0.5949, -0.4746, 0.0000],
        [-0.5359, -1.1236, 0.8792],
        [-0.5359, -1.1236, -0.8792],
        [0.5949, 0.4746, 0.0000],
        [0.5359, 1.1236, 0.8792],
        [0.5359, 1.1236, -0.8792],
        [1.9264, -0.2646, 0.0000],
        [2.7440, 0.4570, 0.0000],
        [2.0324, -0.8974, 0.8842],
        [2.0324, -0.8974, -0.8842],
    ]
)


def main() -> None:
    s = Structure(
        names=NAMES,
        resnames=["BUT"] * 14,
        resids=[1] * 14,
        elements=ELEMENTS,
        coords=COORDS_A * 0.1,
        title="n-butane all-trans (hand-authored fixture)",
        box=np.array([3.0, 3.0, 3.0]),
    )
    write_multimodel_pdb("fixtures/butane/butane.pdb", s, s.coords[None, :, :])
    lines = [s.title, f"{s.natoms:5d}"]
    for i in range(s.natoms):
        x, y, z = s.coords[i]
        lines.append(
            f"{s.resids[i]:5d}{s.resnames[i]:<5s}{s.names[i]:>5s}{i + 1:5d}"
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
        )
    lines.append(f"{s.box[0]:10.5f}{s.box[1]:10.5f}{s.box[2]:10.5f}")
    with open("fixtures/butane/butane.gro", "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
