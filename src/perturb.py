#perturb.py


def pert_circuit(qc, B_pert, N_sites, alpha):
    print("---- PERTURB DEBUG ----")
    print(f"B_pert = {B_pert}")
    print(f"alpha = {alpha}")

    for site in range(N_sites):
        print(f"PERT site={site+1}: coef={B_pert[0]}, op=Sx, local_angle={alpha * B_pert[0]}")
        print(f"PERT site={site+1}: coef={B_pert[1]}, op=Sy, local_angle={alpha * B_pert[1]}")
        print(f"PERT site={site+1}: coef={B_pert[2]}, op=Sz, local_angle={alpha * B_pert[2]}")
        print(f"PERT combined: site={site+1}, angle_factor={alpha}")

        qc.rx(alpha * B_pert[0], site)
        qc.ry(alpha * B_pert[1], site)
        qc.rz(alpha * B_pert[2], site)