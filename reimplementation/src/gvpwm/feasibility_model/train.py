"""
This module implements
This module implements the FeasibilityModel and ResidualFeasibilityModel for
learning soft feasibility penalties in the GVP-WM framework.

FeasibilityModel learns epsilon_theta(z_{t+1}^k, h_t, a_t, k), which is used
to compute a soft feasibility penalty P_theta = ||epsilon_theta(.)||^2 which
replaces hard constraints enforced by ALM with a learned energy term.

ResidualFeasibilityModel is a diagnostic control trained that learns by
training on residuals delta_t = z_{t+1} - f_psi(h_t, a_t).
"""
