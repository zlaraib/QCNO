#!/usr/bin/env python
# coding: utf-8
#
# Unit test for the exact-state observable recorders in src/observables.py. It runs
# no circuit: it hands the recorders states whose entanglement and magic are known
# analytically and checks what they write. Run from the repo root:
#     python tests/test_observables.py
#
# NOTE: the expected M_2 of every *subsystem* here is 1, not 0, even though GHZ is a
# stabilizer state, because stabilizer_renyi_magic_from_rdm applies the pure-state
# formula to a mixed RDM and so measures mixedness too (see the TODO in
# entanglement.py). If that normalization is fixed, the subsystem magic expectations
# in test_single_site, test_two_site and test_bipartite all become 0. The global
# magic, computed on the pure full state, is unaffected.

import os
import shutil
import sys
import tempfile

import numpy as np

# Determine the correct path for the 'src' directory
if "__file__" in globals():
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
else:
    src_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit_aer import AerSimulator

from observables import (
    SingleSiteEntanglementObservable,
    TwoSiteEntanglementObservable,
    BipartiteEntanglementObservable,
    GlobalMagicObservable,
)

N_sites = 4
tolerance = 1e-8


def ghz_state():
    """|GHZ> = (|0000> + |1111>)/sqrt(2). Every proper subsystem is an even mixture
    of two orthogonal states, so its purity is 1/2 and S_2 = 1 bit, no matter how
    many sites are kept. GHZ is a stabilizer state, so its magic is 0."""
    amplitudes = np.zeros(2 ** N_sites)
    amplitudes[0] = amplitudes[-1] = 1 / np.sqrt(2)
    return DensityMatrix(Statevector(amplitudes))


def product_state():
    """|0000>: a product stabilizer state, so every entropy and every magic is 0."""
    amplitudes = np.zeros(2 ** N_sites)
    amplitudes[0] = 1.0
    return DensityMatrix(Statevector(amplitudes))


def make_state(t, direct_state):
    """The part of the driver's per-step state dict these recorders read."""
    return {
        "t": t,
        "direct_state": direct_state,
        "particle_ids": np.arange(N_sites),
    }


def load_rows(datadir, name):
    return np.atleast_2d(np.loadtxt(os.path.join(datadir, name)))


def record(observable, datadir):
    """Run one recorder over the GHZ state then the product state, and return the
    progress values of each step."""
    params = {
        "datadir": datadir,
        "N_sites": N_sites,
        "backend": AerSimulator(method="density_matrix"),
    }
    observable.create_file(params)
    return [
        observable.append_row(params, make_state(0.0, ghz_state())),
        observable.append_row(params, make_state(1.0, product_state())),
    ]


def test_single_site(datadir):
    """GHZ: every site is maximally mixed -> S_2 = 1. Its M_2 also comes out as 1:
    on a mixed rho this functional is measuring mixedness, not non-stabilizerness
    (see the TODO in entanglement.py)."""
    ghz, product = record(SingleSiteEntanglementObservable(), datadir)
    assert np.allclose(ghz, [1.0, 1.0], atol=tolerance), ghz
    assert np.allclose(product, [0.0, 0.0], atol=tolerance), product

    renyi2 = load_rows(datadir, "t_renyi2_single_direct.dat")
    magic = load_rows(datadir, "t_magic_single_direct.dat")
    assert renyi2.shape == (2, N_sites + 1), renyi2.shape
    assert magic.shape == (2, N_sites + 1), magic.shape

    # column 0 is the time, the rest are one value per particle
    assert np.allclose(renyi2[:, 0], [0.0, 1.0], atol=tolerance), renyi2
    assert np.allclose(renyi2[0, 1:], 1.0, atol=tolerance), renyi2
    assert np.allclose(magic[0, 1:], 1.0, atol=tolerance), magic
    assert np.allclose(renyi2[1, 1:], 0.0, atol=tolerance), renyi2
    assert np.allclose(magic[1, 1:], 0.0, atol=tolerance), magic

    print("SingleSiteEntanglementObservable: passed")


def test_two_site(datadir):
    """Same GHZ values for every pair, and the pair key lists all i < j."""
    ghz, product = record(TwoSiteEntanglementObservable(), datadir)
    assert np.allclose(ghz, [1.0, 1.0], atol=tolerance), ghz
    assert np.allclose(product, [0.0, 0.0], atol=tolerance), product

    pairs = load_rows(datadir, "two_site_subsets.dat")
    expected_pairs = [(i, j) for i in range(N_sites) for j in range(i + 1, N_sites)]
    assert np.array_equal(pairs, expected_pairs), pairs

    renyi2 = load_rows(datadir, "t_renyi2_two_site_direct.dat")
    magic = load_rows(datadir, "t_magic_two_site_direct.dat")
    assert renyi2.shape == (2, len(expected_pairs) + 1), renyi2.shape
    assert np.allclose(renyi2[0, 1:], 1.0, atol=tolerance), renyi2
    assert np.allclose(magic[0, 1:], 1.0, atol=tolerance), magic
    assert np.allclose(renyi2[1, 1:], 0.0, atol=tolerance), renyi2
    assert np.allclose(magic[1, 1:], 0.0, atol=tolerance), magic

    print("TwoSiteEntanglementObservable: passed")


def test_bipartite(datadir):
    """The 2|2 bipartition of GHZ has the same purity 1/2 -> S_2 = 1."""
    ghz, product = record(BipartiteEntanglementObservable(), datadir)
    assert np.allclose(ghz, [1.0, 1.0], atol=tolerance), ghz
    assert np.allclose(product, [0.0, 0.0], atol=tolerance), product

    renyi2 = load_rows(datadir, "t_renyi2_bipartite_direct.dat")
    magic = load_rows(datadir, "t_magic_bipartite_direct.dat")
    assert renyi2.shape == (2, 2), renyi2.shape  # one scalar column per timestep
    assert np.allclose(renyi2[:, 1], [1.0, 0.0], atol=tolerance), renyi2
    assert np.allclose(magic[:, 1], [1.0, 0.0], atol=tolerance), magic

    print("BipartiteEntanglementObservable: passed")


def test_global_magic(datadir):
    """Both states are stabilizer states, so the global magic is 0 for both. Both
    are also pure, so their Pauli weight distributions sum to 1."""
    ghz, product = record(GlobalMagicObservable(), datadir)
    assert np.allclose(ghz, [0.0], atol=tolerance), ghz
    assert np.allclose(product, [0.0], atol=tolerance), product

    weights = load_rows(datadir, "t_pauli_weight_distribution_direct.dat")
    assert weights.shape == (2, N_sites + 2), weights.shape  # time + weights 0..N
    assert np.allclose(weights[:, 1:].sum(axis=1), 1.0, atol=tolerance), weights

    # GHZ has weight-0, weight-2 and weight-4 Pauli support only; |0000> is built
    # from Z strings of every weight
    assert np.allclose(weights[0, 2::2], 0.0, atol=tolerance), weights

    sectors = load_rows(datadir, "pauli_weights.dat")
    assert np.array_equal(sectors.ravel(), np.arange(N_sites + 1)), sectors

    print("GlobalMagicObservable: passed")


datadir = tempfile.mkdtemp(prefix="qcno_test_observables_")
try:
    test_single_site(datadir)
    test_two_site(datadir)
    test_bipartite(datadir)
    test_global_magic(datadir)
    print("Test passed.")
finally:
    shutil.rmtree(datadir, ignore_errors=True)
