from pathlib import Path

import numpy as np
import pytest

from ffrefine.config import Binaries
from ffrefine.errors import ExternalToolError
from ffrefine.io.structure import read_pdb
from ffrefine.qm.orca import (
    orca_env,
    parse_final_energy,
    refine_energies,
    write_sp_input,
)

SAMPLE_OUT = """
------------------- FINAL SINGLE POINT ENERGY -------------------
FINAL SINGLE POINT ENERGY        -1.116759307204
"""


def test_parse_final_energy_reads_the_hartree_value():
    assert parse_final_energy(SAMPLE_OUT) == pytest.approx(-1.116759307204)


def test_parse_final_energy_takes_the_last_when_several_appear():
    """A geometry optimisation prints one per cycle; the converged one is last."""
    text = "FINAL SINGLE POINT ENERGY -1.0\nFINAL SINGLE POINT ENERGY -2.0\n"
    assert parse_final_energy(text) == pytest.approx(-2.0)


def test_parse_final_energy_refuses_output_without_one():
    with pytest.raises(ExternalToolError, match="FINAL SINGLE POINT ENERGY"):
        parse_final_energy("SCF NOT CONVERGED\n")


def test_sp_input_carries_keywords_charge_and_multiplicity(tmp_path, butane_pdb):
    s = read_pdb(butane_pdb)
    text = write_sp_input(
        tmp_path / "sp.inp", s, s.coords, "! HF STO-3G", charge=0, mult=1, nprocs=2
    )
    assert text.startswith("! HF STO-3G")
    assert "%pal nprocs 2" in text
    assert "*xyz 0 1" in text
    assert text.rstrip().endswith("*")


def test_sp_input_omits_the_pal_block_when_single_threaded(tmp_path, butane_pdb):
    """A %pal block of one is noise, and ORCA is happier without it."""
    s = read_pdb(butane_pdb)
    text = write_sp_input(tmp_path / "sp.inp", s, s.coords, "! HF STO-3G", 0, 1, nprocs=1)
    assert "%pal" not in text


def test_sp_input_states_the_multiplicity_it_was_given(tmp_path, butane_pdb):
    """A triplet written as a singlet is a different calculation, silently."""
    s = read_pdb(butane_pdb)
    text = write_sp_input(tmp_path / "sp.inp", s, s.coords, "! HF STO-3G", charge=-1, mult=3)
    assert "*xyz -1 3" in text


def test_sp_input_is_in_angstrom(tmp_path, butane_pdb):
    s = read_pdb(butane_pdb)
    text = write_sp_input(tmp_path / "sp.inp", s, s.coords, "! HF STO-3G", 0, 1)
    first = [ln for ln in text.splitlines() if ln.startswith(("C ", "H "))][0]
    assert float(first.split()[1]) == pytest.approx(s.coords[0][0] * 10.0, abs=1e-5)


def test_orca_env_puts_the_libdir_on_ld_library_path(tmp_path):
    """A shared ORCA build will not start otherwise."""
    env = orca_env(Binaries(orca=str(tmp_path / "orca"), orca_libdir=str(tmp_path)))
    assert str(tmp_path) in env["LD_LIBRARY_PATH"]


def test_orca_env_defaults_the_libdir_to_the_binarys_own_directory(tmp_path):
    env = orca_env(Binaries(orca=str(tmp_path / "orca")))
    assert str(tmp_path) in env["LD_LIBRARY_PATH"]


@pytest.mark.needs_orca
def test_refine_energies_runs_real_orca(tmp_path, butane_pdb, orca_bin):
    """Cheap keywords on purpose: the real default is 24 frames of hybrid DFT."""
    s = read_pdb(butane_pdb)
    frames = np.stack([s.coords, s.coords])
    energies = refine_energies(
        Binaries(orca=orca_bin), s, frames, tmp_path, "! HF STO-3G", charge=0, mult=1
    )
    assert energies.shape == (2,)
    # Same geometry twice, so the two energies must agree exactly.
    assert energies[0] == pytest.approx(energies[1], rel=1e-10)
    # Butane at HF/STO-3G is around -155 Hartree, i.e. very negative in kJ/mol.
    assert energies[0] < -300000.0


ORCA_FIXTURES = Path(__file__).parent.parent / "fixtures" / "butane" / "orca"


def test_parse_chelpg_charges_reads_all_fourteen():
    from ffrefine.qm.orca import parse_chelpg_charges

    q = parse_chelpg_charges(ORCA_FIXTURES / "esp.chelpg.xyz")

    assert q.shape == (14,)
    assert q[0] == pytest.approx(-0.40576138)
    assert q[10] == pytest.approx(-0.40905194)
    assert q.sum() == pytest.approx(0.0, abs=1e-6)


def test_parse_chelpg_charges_refuses_a_file_with_no_charge_column(tmp_path):
    from ffrefine.qm.orca import parse_chelpg_charges

    path = tmp_path / "bad.xyz"
    path.write_text("1\ncomment\nC 0.0 0.0 0.0\n")

    with pytest.raises(ExternalToolError, match="charge"):
        parse_chelpg_charges(path)


def test_read_chelpg_vpot_takes_potential_from_the_first_column():
    """The CHELPG .vpot is V x y z. Reading it as x y z V silently swaps them."""
    from ffrefine.qm.orca import read_chelpg_vpot

    grid_nm, v = read_chelpg_vpot(ORCA_FIXTURES / "esp_head.vpot")

    assert grid_nm.shape == (5, 3)
    assert v.shape == (5,)
    assert v[0] == pytest.approx(1.9701622e-04)
    assert v[4] == pytest.approx(-1.2312459e-04)
    # -9.9097238 bohr is about -0.5244 nm; if the columns were swapped this
    # would be the 1.97e-04 potential instead.
    assert grid_nm[0][0] == pytest.approx(-0.52440, abs=1e-4)


def test_read_chelpg_vpot_refuses_a_truncated_file(tmp_path):
    from ffrefine.qm.orca import read_chelpg_vpot

    path = tmp_path / "short.vpot"
    path.write_text("   2  3\n 0.0 0.0 0.0\n 0.0 0.0 0.0\n 1.0 2.0 3.0 4.0\n")

    with pytest.raises(ExternalToolError, match="3"):
        read_chelpg_vpot(path)


def test_esp_input_adds_chelpg_to_the_keyword_line(tmp_path, butane_pdb):
    """--orca-esp-keywords carries the level only; the scheme adds CHELPG."""
    from ffrefine.io.structure import read_pdb
    from ffrefine.qm.orca import esp_keyword_line

    assert esp_keyword_line("! HF 6-31G*", chelpg=True) == "! HF 6-31G* CHELPG"
    assert esp_keyword_line("! HF 6-31G*", chelpg=False) == "! HF 6-31G*"


def test_esp_keyword_line_does_not_double_chelpg():
    """A user who passes CHELPG themselves must not get it twice."""
    from ffrefine.qm.orca import esp_keyword_line

    assert esp_keyword_line("! HF 6-31G* CHELPG", chelpg=True) == "! HF 6-31G* CHELPG"


@pytest.mark.needs_orca
def test_run_esp_job_produces_the_files_the_schemes_need(tmp_path, butane_pdb, orca_bin):
    """Cheap keywords on purpose; the default is hybrid DFT."""
    from ffrefine.config import Binaries
    from ffrefine.io.structure import read_pdb
    from ffrefine.qm.orca import parse_chelpg_charges, run_esp_job

    s = read_pdb(butane_pdb)
    base = run_esp_job(
        Binaries(orca=orca_bin), s, s.coords, tmp_path, "! HF STO-3G", charge=0, mult=1
    )

    assert base.with_suffix(".gbw").exists()
    assert Path(str(base) + ".chelpg.xyz").exists()
    assert Path(str(base) + ".vpot").exists()
    q = parse_chelpg_charges(Path(str(base) + ".chelpg.xyz"))
    assert q.shape == (14,)
    assert q.sum() == pytest.approx(0.0, abs=1e-4)
