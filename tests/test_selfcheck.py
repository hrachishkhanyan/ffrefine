import numpy as np
import pytest
from ffrefine.errors import ValidationError
from ffrefine.io.structure import read_pdb
from ffrefine.io.topology import read_topology
from ffrefine.mm import selfcheck


@pytest.mark.needs_gmx
def test_self_consistency_passes_on_butane(
    tmp_path, butane_pdb, butane_itp, gmx_bin, rotated_frames
):
    """Spec section 4.5: agreement better than 1e-3 kJ/mol at every point."""
    from ffrefine.config import Binaries

    s = read_pdb(butane_pdb)
    t = read_topology(butane_itp)
    frames = rotated_frames(s, 12)
    r = selfcheck.check_torsion_removal(
        Binaries(gmx=gmx_bin), t, (1, 5, 8, 11), s, frames, tmp_path
    )
    assert r.passed
    assert r.max_abs_error < 1e-3


@pytest.mark.needs_gmx
def test_self_check_samples_a_real_torsion_range(
    tmp_path, butane_pdb, butane_itp, gmx_bin, rotated_frames
):
    from ffrefine.config import Binaries

    s = read_pdb(butane_pdb)
    t = read_topology(butane_itp)
    frames = rotated_frames(s, 12)
    r = selfcheck.check_torsion_removal(
        Binaries(gmx=gmx_bin), t, (1, 5, 8, 11), s, frames, tmp_path
    )
    assert r.phi_deg.ptp() > 180.0
    assert np.abs(r.analytic_difference).max() > 0.5


@pytest.mark.needs_gmx
def test_self_check_raises_when_tolerance_impossible(
    tmp_path, butane_pdb, butane_itp, gmx_bin, rotated_frames
):
    """A deliberately absurd tolerance must abort rather than warn."""
    from ffrefine.config import Binaries

    s = read_pdb(butane_pdb)
    t = read_topology(butane_itp)
    frames = rotated_frames(s, 6)
    with pytest.raises(ValidationError, match="self-consistency"):
        selfcheck.check_torsion_removal(
            Binaries(gmx=gmx_bin),
            t,
            (1, 5, 8, 11),
            s,
            frames,
            tmp_path,
            tolerance=1e-18,
        )
