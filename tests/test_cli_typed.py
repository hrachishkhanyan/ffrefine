"""The CLI on a real CGenFF topology: typed torsions and #include'd force fields.

Milestone 6a. The unit tests here stub the GROMACS-facing calls, so they need
no binary; the live tests are marked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from ffrefine import cli
from ffrefine.cli import main
from ffrefine.io.includes import MARKER_NOTE, MARKER_PREFIX
from ffrefine.io.topwriter import write_refit_topology
from ffrefine.mm.validate_fit import ValidationResult
from ffrefine.stages import hash_array

FIXTURES = Path(__file__).parent.parent / "fixtures"
PFOA = FIXTURES / "pfoa_gromacs"
PFOA_QUAD = (25, 1, 3, 4)
PFOA_FF = (PFOA / "charmm36.ff").resolve()


def _pfoa_args(outdir: Path, top: Path | None = None) -> list[str]:
    return [
        "-s",
        str(PFOA / "pfoa_gmx.pdb"),
        "-p",
        str(top or PFOA / "pfoa_gmx.top"),
        "--dihedral",
        ",".join(map(str, PFOA_QUAD)),
        "--outdir",
        str(outdir),
    ]


def _include_lines(path: Path) -> list[str]:
    return [
        line for line in Path(path).read_text().splitlines() if re.match(r"^\s*#\s*include", line)
    ]


def _marker(path: Path) -> list[str]:
    return Path(path).read_text().splitlines()[:2]


def _write_pfoa_stage_inputs(outdir: Path, n: int = 24) -> None:
    """Hand-built 01_qm_scan and 02_mm_scan payloads for PFOA (see test_cli.py).

    ``fit`` only hashes the frames, so placeholder coordinates of the right
    shape are enough, and they must be the same in both payloads.
    """
    phi = np.linspace(-180.0, 180.0, n, endpoint=False)
    qm = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    frames = np.arange(n * 26 * 3, dtype=float).reshape(n, 26, 3) * 0.01
    (outdir / "01_qm_scan").mkdir(parents=True, exist_ok=True)
    (outdir / "01_qm_scan" / "scan.json").write_text(
        json.dumps(
            {"phi_deg": phi.tolist(), "energy_kj": qm.tolist(), "frames_nm": frames.tolist()}
        )
    )
    (outdir / "02_mm_scan").mkdir(parents=True, exist_ok=True)
    (outdir / "02_mm_scan" / "energies.json").write_text(
        json.dumps(
            {
                "quad": list(PFOA_QUAD),
                "frames_sha256": hash_array(frames),
                "phi_deg": phi.tolist(),
                "zeroed_potential_kj": [0.0] * n,
                "full_potential_kj": [0.0] * n,
            }
        )
    )


def _fake_validate_refit(
    binaries, topology, quad, structure, frames, qm, k, delta, mult, workdir, **kwargs
):
    """validate_refit without GROMACS: writes after.itp the way the real one does."""
    write_refit_topology(
        topology,
        quad,
        k,
        delta,
        mult,
        Path(workdir) / "after.itp",
        distribute=kwargs.get("distribute", False),
        includes="absolute",
    )
    n = len(qm)
    return ValidationResult(
        rmse_before=1.0,
        rmse_after=0.5,
        mm_before_kj=np.zeros(n),
        mm_after_kj=np.zeros(n),
        qm_kj=np.asarray(qm, dtype=float),
        phi_deg=np.linspace(-180.0, 180.0, n, endpoint=False),
        improved=True,
    )


def _stub_validate(monkeypatch) -> None:
    """Replace every GROMACS-facing call cmd_validate makes."""
    monkeypatch.setattr(cli, "validate_refit", _fake_validate_refit)


def test_fit_writes_dihedral_itp_with_the_users_includes(tmp_path):
    out = tmp_path / "out"
    _write_pfoa_stage_inputs(out)
    assert main(["fit"] + _pfoa_args(out)) == 0
    itp = out / "03_fit" / "dihedral.itp"
    assert _include_lines(itp) == _include_lines(PFOA / "pfoa_gmx.top")
    assert _marker(itp) == [MARKER_NOTE, f"{MARKER_PREFIX}{PFOA.resolve()}"]


def test_validate_writes_molecule_refined_as_a_drop_in(tmp_path, monkeypatch):
    """final/after.itp goes to grompp (absolute); molecule_refined.itp is kept (verbatim)."""
    out = tmp_path / "out"
    _write_pfoa_stage_inputs(out)
    assert main(["fit"] + _pfoa_args(out)) == 0
    _stub_validate(monkeypatch)
    assert main(["validate"] + _pfoa_args(out)) == 0
    final = out / "final"
    refined = final / "molecule_refined.itp"
    assert _include_lines(refined) == _include_lines(PFOA / "pfoa_gmx.top")
    assert _marker(refined) == [MARKER_NOTE, f"{MARKER_PREFIX}{PFOA.resolve()}"]
    assert f'#include "{PFOA_FF / "forcefield.itp"}"' in _include_lines(final / "after.itp")
