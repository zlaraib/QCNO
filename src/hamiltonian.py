#hamiltonian.py
import numpy as np
from momentum import momentum
from constants import G_F
from geometric_func import geometric_func
from shape_func import shape_func

def construct_hamiltonian(params, state):

    """
    Build the Hamiltonian as a list of (coefficient, Pauli) terms.

    Inputs:
    - params: run-parameter dict.
    - state: the driver's per-step dict, for the occupancies, positions, momenta
      and vacuum frequencies as they stand at this substep.

    Output:
    - terms: List of (coefficient, operator, qubits) tuples, where each term is one of:
        * (coef, 'JJ', (q0, q1))          isotropic two-site coupling on qubits
                                          q0, q1, i.e. coef * (X⊗X + Y⊗Y + Z⊗Z)
        * (coef_vec, 'BJ', (q,))          full single-site (vacuum) term on qubit q,
                                          coef_vec = (omega/2)*B = per-axis (cx,cy,cz)
                                          for cx*X + cy*Y + cz*Z
      All 'JJ' terms come first (as a generalized even/odd brickwork over spacing),
      followed by one 'BJ' term per site. Qubit indices already include the
      site->qubit reversal (site i maps to qubit N_sites - 1 - i) that the old
      big-endian Pauli labels encoded, so downstream code applies gates directly to
      these indices with no remapping.
    """
    terms = []

    p_mod, p_hat = momentum(state["p"], params["N_sites"])

    def q(site):
        # Preserve the big-endian convention of the old Pauli labels:
        # site placed at label position `site` acted on qubit N_sites - 1 - site.
        return params["N_sites"] - 1 - site

    # Two-site coupling terms, emitted as a generalized even/odd brickwork.
    #
    # The loop still covers ALL pairs (all-to-all): the outer index `d` is the
    # spacing j - i, so every pair (i, i+d) with i < j appears exactly once. It
    # does NOT assume nearest-neighbor connectivity -- longer-range couplings just
    # come out with whatever strength geometric_func/shape_func assign (periodicity
    # is baked into those strengths, not into this loop).
    #
    # For each spacing d, the bonds are split into two sub-layers by
    # parity = (i // d) % 2. Within a sub-layer no two bonds share a qubit
    # (same-color bonds never differ by exactly d), so those gates commute and
    # their ordering contributes no Trotter error -- only the inter-layer
    # non-commutativity remains. For d == 1 this is the standard nearest-neighbor
    # even/odd: bonds (0,1),(2,3),... then (1,2),(3,4),...
    for d in range(1, params["N_sites"]):
        for parity in (0, 1):
            for i in range(params["N_sites"] - d):
                if (i // d) % 2 != parity:
                    continue
                j = i + d
                geometric_factor = geometric_func(params["geometric_name"], p_hat, i, j)
                shape_function = shape_func(state["x"], params["dp"], i, j, params["L"], params["shape_name"], params["periodic"])
                interaction_strength = ((1/2) * np.sqrt(2) * G_F * (state["N"][i] + state["N"][j]) / (2 * ((params["dx"])**3))) * geometric_factor * shape_function

                # print("geometric_factor from site ", i, " and site ", j, "= ", geometric_factor)
                # print("shape_function from site ", i, " and site ", j, "= ", shape_function)
                # print("interaction_strength from site ", i, " and site ", j, "= ", interaction_strength)

                if interaction_strength != 0:
                    terms.append((interaction_strength, 'JJ', (q(i), q(j))))

    # Single-site (vacuum) terms: one grouped 'BJ' term per site, appended after
    # all two-site 'JJ' terms. Each carries the full per-axis coefficient vector
    # (omega[k]/2) * B, emitted downstream (evolve.apply_single_site_gate) as one
    # exact 1-qubit UnitaryGate. Because the site's whole field lives in a single
    # term, moving to a per-site field later is just indexing B by site here.
    for k in range(params["N_sites"]):
        if state["omega"][k] != 0:
            terms.append(((state["omega"][k] / 2) * np.array(params["B"]), 'BJ', (q(k),)))

    return terms