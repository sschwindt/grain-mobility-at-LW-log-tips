#!/usr/bin/env python3
"""
Compute discharge, depth, and mean velocity needed to mobilize grains at a log tip according to:
    * Schalko, Ponce et al. (2024) Flow and Turbulence Due To Wood Contribute to Declogging of Gravel Bed. GRL 51.
        https://doi.org/10.1029/2023GL107507
    * Schalko, Follett, Nepf (2023) Impact of Lateral Gap on Flow Distribution, Backwater Rise, and Turbulence Generated
        by a Logjam. Water Resources Research 59:e2023WR034689. https://doi.org/10.1029/2023WR034689
    * Schalko, Wohl, Nepf (2021) Flow and wake characteristics associated with large wood to inform river restoration.
        Sci Rep 11:8644. https://doi.org/10.1038/s41598-021-87892-7

Assumptions (for lab flume design):
    * Uniform flow in the approach reach so energy slope equals bed slope
    * Rectangular cross-section of width w_channel
    * A single calibrated Darcy Weissbach style friction coefficient Cf (Schalko et al. 2023 / Julien 2010)
    * Local tip velocity is a multiplier of the approach mean velocity u_tip = multiplier * U_mean, where the multiplier
        range is based on Schalko et al. (2023, 2024): 1.7--2.0

Mobility criterion:
tau = Cf * (u_tip^2 + beta * k_t) / ((s - 1) * g * d50)
  where k_t = gamma * u_tip^2; that is, turbulence is modeled with factor (1 + beta * gamma);
  set gamma to 0.0 to use a pure mean velocity threshold with the multiplier (1.7).

Notes:
    * This script computes approach depth and discharge under a uniform flow closure using Cf from the reference state.
    * It does not resolve backwater or head loss induced by the log, so results are just first-order design estimates.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path

# DEFINE USER PARAMETERS HERE --- --- --- --- --- --- --- --- ---- --- --- --- --- --- --- --- --- --- --- --- --- --- -
# HYDRAULIC ENVIRONMENT
w_channel = 0.25  # channel width (m)
S = 0.003  # longitudinal channel slope (-)
d50 = 0.008  # present surface bed grain diameter (m)
# LOG GEOMETRY
log_diameter = 0.08
log_length = 0.10
w_gap = w_channel - log_length
if w_gap <= 0:
    raise ValueError("Log length must be smaller than channel width.")
# MULTIPLIER FOR TIP OR NEAR TIP SPEED RELATIVE TO MEAN (Schalko et al. 2024)
multiplier = 1.7


class TeeStream:
    """Log (print) to multiple streams; that is, mirror stdout and stderr to a logfile and console output.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def hydraulic_radius_rectangular(B: float, h: float) -> float:
    A = B * h
    P = B + 2.0 * h
    return A / P


def get_Cf(B: float, h_ref: float, S: float, Q_ref: float, g: float) -> float:
    """
    Calibrate Cf from a reference uniform flow condition in the unobstructed flume.

    u_star = sqrt(g * R * S)
    U_ref = Q_ref / (B * h_ref)
    Cf = (u_star / U_ref)^2
    """
    if B <= 0 or h_ref <= 0 or S <= 0 or Q_ref <= 0:
        raise ValueError("B, h_ref, S, Q_ref must be positive for Cf calibration.")
    R_ref = hydraulic_radius_rectangular(B, h_ref)
    u_star = math.sqrt(g * R_ref * S)
    U_ref = Q_ref / (B * h_ref)
    return (u_star / U_ref) ** 2


def get_u_from_h(B: float, h: float, S: float, g: float, Cf: float) -> float:
    """
    Uniform flow closure using Cf and u_star.

    u_star^2 = g * R * S
    U = u_star / sqrt(Cf) = sqrt(g * R * S / Cf)
    """
    if B <= 0 or h <= 0 or S <= 0 or g <= 0 or Cf <= 0:
        raise ValueError("B, h, S, g, Cf must be positive.")
    R = hydraulic_radius_rectangular(B, h)
    return math.sqrt(g * R * S / Cf)


def solve_h_for_U(B: float, S: float, g: float, Cf: float, U_target: float) -> float:
    """
    Solve for h such that U_uniform_from_h equals U_target using bisection.

    There is a hard upper bound as h goes to infinity:
    R -> B/2 so U_max = sqrt(g * (B/2) * S / Cf)
    """
    if U_target <= 0:
        raise ValueError("U_target must be positive.")

    U_max = math.sqrt(g * (B / 2.0) * S / Cf)
    if U_target >= U_max:
        raise ValueError(
            f"U_target = {U_target:.4f} m/s exceeds U_max = {U_max:.4f} m/s under this uniform flow model."
        )

    h_low = 1e-6
    h_high = B
    while get_u_from_h(B, h_high, S, g, Cf) < U_target:
        h_high *= 2.0
        if h_high > 100.0 * B:
            raise ValueError("Failed to bracket a solution for h. Check inputs.")

    for _ in range(80):
        h_mid = 0.5 * (h_low + h_high)
        U_mid = get_u_from_h(B, h_mid, S, g, Cf)
        if U_mid < U_target:
            h_low = h_mid
        else:
            h_high = h_mid

    return 0.5 * (h_low + h_high)


def get_u_tip(tau_cr: float, s_rel: float, g: float, d50: float, Cf: float, beta: float, gamma: float) -> float:
    """
    u_tip_crit from tau_cr with turbulence closure k_t = gamma * u_tip^2.

    tau_cr = Cf * u_tip^2 * (1 + beta*gamma) / ((s - 1) * g * d50)
    """
    if tau_cr <= 0 or s_rel <= 1 or g <= 0 or d50 <= 0 or Cf <= 0:
        raise ValueError("tau_cr, s_rel, g, d50, Cf must be valid and positive.")
    factor = 1.0 + beta * gamma
    if factor <= 0:
        raise ValueError("1 + beta*gamma must be positive.")
    return math.sqrt(tau_cr * (s_rel - 1.0) * g * d50 / (Cf * factor))


def compute_conditions_for_multiplier(
    B: float,
    S: float,
    g: float,
    Cf: float,
    tau_cr: float,
    rho: float,
    rho_s: float,
    d50: float,
    beta: float,
    gamma: float,
    multiplier: float,
) -> dict:
    if multiplier <= 0:
        raise ValueError("multiplier must be positive.")

    s_rel = rho_s / rho
    u_tip_crit = get_u_tip(tau_cr, s_rel, g, d50, Cf, beta, gamma)

    U_required = u_tip_crit / multiplier
    h_required = solve_h_for_U(B, S, g, Cf, U_required)
    Q_required = U_required * B * h_required

    Fr = U_required / math.sqrt(g * h_required)

    return {
        "multiplier": multiplier,
        "u_tip_crit_ms": u_tip_crit,
        "U_mean_required_ms": U_required,
        "h_required_m": h_required,
        "Q_required_m3s": Q_required,
        "Q_required_Ls": 1000.0 * Q_required,
        "Froude": Fr,
    }


def flume_design_processor() -> None:
    # TURBULENCE AUGMENTATION
    beta = 4.0      # empirical scaling factor from Schalko et al. (2024); plus-minus 0.4
    gamma = 0.0     # if available: measured kt divided by u_tip**2

    # CONSTANTS
    rho = 1000.0      # water density (kg/m3)
    rho_s = 2650.0    # sediment density (kg/m3)
    g = 9.81          # gravity
    tau_cr = 0.047   # critical Shields parameter (--)

    # REFERENCE MOBILITY CONDITION TO CALIBRATE Cf IN THE UNOBSTRUCTED FLUME
    h_ref = tau_cr * (rho_s / rho - 1) * d50 / S # critical water depth for incipient motion
    n_m = 1 / (26 / (d50 ** (1 / 6)))            # Manning n (MPM 1948)
    Q_ref = 1 / n_m * math.sqrt(S) * (w_channel * h_ref / (w_channel+2*h_ref)) ** (2 / 3) * w_channel * h_ref # Q crit
    Cf = get_Cf(w_channel, h_ref, S, Q_ref, g)

    print("FLUME SETUP")
    print(f" > Channel width = {w_channel:.3f} m")
    print(f" > Channel slope = {S:.5f}")
    print(f" > Grain d50 = {d50:.4f} m")
    print(f" > Log diameter = {log_diameter:.3f} m")
    print(f" > Log length = {log_length:.3f} m")
    print(f" > Gap width (log tip to other wall) = {w_gap:.3f} m")
    print(f" > Cf calibrated from reference mobility discharge = {Q_ref:.3f} m3/s at h = {h_ref:.3f} m >> Cf = {Cf:.6f}")
    print(f" > Empirical turbulence calibration coefficient (Schalko et al. 2024): beta = {beta:.2f}, gamma = {gamma:.3f}")

    out = compute_conditions_for_multiplier(
        B=w_channel, S=S, g=g, Cf=Cf,
        tau_cr=tau_cr, rho=rho, rho_s=rho_s, d50=d50,
        beta=beta, gamma=gamma, multiplier=multiplier
    )

    print("\n\nCONDITIONS FOR INCIPIENT GRAIN MOTION AT THE LOG TIP")
    print(f" > Velocity multiplier (minimum from Schalko et al. 2024) = {out['multiplier']:.2f}")
    print(f" > Critical velocity for grain mobility at the log tip = {out['u_tip_crit_ms']:.4f} m/s")
    print(f" > Bulk flow velocity U (without log) = {out['U_mean_required_ms']:.4f} m/s")
    print(f" > Bulk water depth h (without log) = {out['h_required_m']:.4f} m")
    print(f" > Required discharge for grain mobility at the log tip Q = {out['Q_required_m3s']:.6f} m3/s  ({out['Q_required_Ls']:.2f} L/s)")
    print(f" > Bulk Froude number = {out['Froude']:.3f}")


if __name__ == "__main__":
    script_path = Path(__file__).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_path = script_path.with_name(f"{script_path.stem}_{timestamp}.log")
    log_path = script_path.with_name(
        f"logL={str(log_length)}-logD={str(log_diameter)}-channelW={str(w_channel)}-uUmult={str(multiplier)}-d50={str(d50)}.log"
    )

    with open(log_path, "w", encoding="utf-8") as log_fh:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = TeeStream(old_stdout, log_fh)
        sys.stderr = TeeStream(old_stderr, log_fh)
        try:
            flume_design_processor()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
