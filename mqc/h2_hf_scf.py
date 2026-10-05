import numpy as np
from pyscf import gto

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
S = mol.intor("int1e_ovlp")
H_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
two_e = mol.intor("int2e")

R = 1.4 # Inter nuclear distance in a.u.
V_NN = 1.0/R # Nuclear potential energy

# Symmetrize the 2 x 2 x 2 x 2 tensor.
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                val = two_e[i, j, k, l]
                if val != 0:
                    two_e[j, i, k, l] = val
                    two_e[i, j, l, k] = val
                    two_e[j, i, l, k] = val
                    two_e[k, l, i, j] = val
                    two_e[l, k, i, j] = val
                    two_e[k, l, j, i] = val
                    two_e[l, k, j, i] = val

# -------------------------------------------------------------
# 2. Symmetric Orthogonalization of the Basis: X = S^(-1/2)
# -------------------------------------------------------------
s_vals, s_vecs = np.linalg.eigh(S)
# X = U * diag(s^(-1/2)) * U.T
X = s_vecs @ np.diag(1.0 / np.sqrt(s_vals)) @ s_vecs.T

# -------------------------------------------------------------
# 3. The Self-Consistent Field (SCF) Iteration Loop
# -------------------------------------------------------------
# Initial guess: zero density matrix (equivalent to F = H_core)
P = np.zeros((2, 2))
E_old = 0.0
tol = 1e-10
max_iter = 50

print(f"{'Iter':<5} {'Electronic Energy':<20} {'Total Energy':<20} {'ΔE':<15}")
print("-" * 62)

for iteration in range(1, max_iter + 1):
    # Step A: Build the Fock matrix: F_mu,nu = H_core + sum_ls P_ls 
    # [ (mu nu | ls) - 0.5 (mu s | l nu) ]
    # Using NumPy einsum for clean tensor contraction
    J = np.einsum('ls,mnls->mn', P, two_e)
    K = np.einsum('ls,mlns->mn', P, two_e)
    F = H_core + J - 0.5 * K

    # Step B: Transform Fock matrix to orthogonal basis: F' = X.T @ F @ X
    F_prime = X.T @ F @ X

    # Step C: Diagonalize F' to get orbital energies (eps) and coefficients (C')
    eps, C_prime = np.linalg.eigh(F_prime)

    # Step D: Back-transform coefficients to original AO basis: C = X @ C'
    C = X @ C_prime

    # Step E: Form new density matrix P for closed-shell (N=2, doubly occupied 
    # orbital 0)
    # P_mu,nu = 2 * sum_{a occupied} C_mu,a * C_nu,a
    P_new = 2.0 * np.outer(C[:, 0], C[:, 0])

    # Step F: Compute total energy: E_tot = 0.5 * Tr[ P (H_core + F) ] + V_NN
    E_elec = 0.5 * np.sum(P_new * (H_core + F))
    E_tot = E_elec + V_NN
    delta_E = abs(E_tot - E_old)

    print(f"{iteration:<5} {E_elec:<20.10f} {E_tot:<20.10f} {delta_E:<15.3e}")

    if delta_E < tol:
        print("\n--> SCF converged successfully!")
        break

    P = P_new
    E_old = E_tot

print(f"Final Ground State Energy: {E_tot:.6f} a.u. (Expected: -1.116714 a.u.)")
print(f"Orbital Energies (epsilon): {eps}")
print(f"Ground-state MO coefficients C:\n{C[:, 0]}")                    
