"""Phenotypic Age (Levine et al., 2018, Aging) - corrected implementation.

Published equation (units the coefficients were fitted in):
    albumin        g/L
    creatinine     umol/L
    glucose        mmol/L
    ln(CRP)        CRP in mg/dL
    lymphocyte     %
    MCV            fL
    RDW            %
    ALP            U/L
    WBC            10^9/L  (= 10^3/uL)
    age            years

    xb   = -19.9067 - 0.0336*alb + 0.0095*creat + 0.1953*glu + 0.0954*ln(CRP)
           - 0.0120*lymph + 0.0268*MCV + 0.3306*RDW + 0.00188*ALP + 0.0554*WBC + 0.0804*age
    M    = 1 - exp( -exp(xb) * (exp(120*g) - 1) / g ),   g = 0.0076927
    PA   = 141.50 + ln( -0.00553 * ln(1 - M) ) / 0.09165

The previous version fed conventional units straight into SI-fitted coefficients
and used 0.090165 instead of 0.09165, which made a healthy 50-year-old score ~123.
"""
import math

GAMMA = 0.0076927
INTERCEPT = -19.9067
PA_OFFSET = 141.50
PA_SCALE = 0.09165
PA_K = 0.00553

# unit systems accepted by the API
CONVENTIONAL = "conventional"   # albumin g/dL, creatinine mg/dL, glucose mg/dL, CRP mg/dL
SI = "si"                       # albumin g/L,  creatinine umol/L, glucose mmol/L, CRP mg/L

# hard plausibility limits in the *equation's own* units. Values outside almost always
# mean a unit mix-up, so they are rejected rather than silently computed.
LIMITS = {
    "albumin_g_l": (10, 60), "creatinine_umol_l": (20, 1300), "glucose_mmol_l": (1.5, 45),
    "crp_mg_dl": (0.0005, 50), "lymphocyte_pct": (1, 95), "mcv_fl": (50, 150),
    "rdw_pct": (8, 35), "alp_u_l": (10, 1500), "wbc_10e9_l": (0.5, 100),
}
# typical adult reference ranges (equation units) - used only for informational notes
TYPICAL = {
    "albumin_g_l": (35, 55), "creatinine_umol_l": (45, 110), "glucose_mmol_l": (3.9, 7.8),
    "crp_mg_dl": (0.0, 1.0), "lymphocyte_pct": (20, 45), "mcv_fl": (80, 100),
    "rdw_pct": (11.5, 14.5), "alp_u_l": (30, 130), "wbc_10e9_l": (4.0, 11.0),
}


def to_equation_units(unit_system, albumin, creatinine, glucose, crp,
                      lymphocyte_pct, mcv_fl, rdw_pct, alp_u_l, wbc_10e9_l):
    """Convert lab values to the units the published coefficients use."""
    if unit_system == CONVENTIONAL:
        alb = albumin * 10.0            # g/dL  -> g/L
        cre = creatinine * 88.4         # mg/dL -> umol/L
        glu = glucose / 18.016          # mg/dL -> mmol/L
        crp_dl = crp                    # mg/dL (already)
    elif unit_system == SI:
        alb, cre, glu = albumin, creatinine, glucose
        crp_dl = crp / 10.0             # mg/L -> mg/dL
    else:
        raise ValueError("unit_system must be 'conventional' or 'si'.")
    return dict(albumin_g_l=alb, creatinine_umol_l=cre, glucose_mmol_l=glu, crp_mg_dl=crp_dl,
                lymphocyte_pct=lymphocyte_pct, mcv_fl=mcv_fl, rdw_pct=rdw_pct,
                alp_u_l=alp_u_l, wbc_10e9_l=wbc_10e9_l)


def phenotypic_age_core(age, albumin_g_l, creatinine_umol_l, glucose_mmol_l, crp_mg_dl,
                        lymphocyte_pct, mcv_fl, rdw_pct, alp_u_l, wbc_10e9_l):
    """Raw equation. Inputs MUST already be in the equation's units."""
    xb = (INTERCEPT - 0.0336 * albumin_g_l + 0.0095 * creatinine_umol_l + 0.1953 * glucose_mmol_l
          + 0.0954 * math.log(crp_mg_dl) - 0.0120 * lymphocyte_pct + 0.0268 * mcv_fl
          + 0.3306 * rdw_pct + 0.00188 * alp_u_l + 0.0554 * wbc_10e9_l + 0.0804 * age)
    xb = max(-50.0, min(50.0, xb))
    # 1 - M = exp(-H)  =>  ln(1 - M) = -H.  Using H directly avoids precision loss at tiny M.
    hazard = math.exp(xb) * (math.exp(120 * GAMMA) - 1) / GAMMA
    mortality = -math.expm1(-hazard)
    pa = PA_OFFSET + math.log(PA_K * hazard) / PA_SCALE
    return pa, mortality, xb


def phenotypic_age(age, albumin, creatinine, glucose, crp, lymphocyte, mcv, rdw, alp, wbc,
                   unit_system=CONVENTIONAL):
    """Validated wrapper. Returns dict with phenotypic_age, mortality_score, notes."""
    raw = [age, albumin, creatinine, glucose, crp, lymphocyte, mcv, rdw, alp, wbc]
    if not all(math.isfinite(float(x)) for x in raw):
        raise ValueError("All inputs must be finite numbers.")
    if min(raw) <= 0:
        raise ValueError("Age and all nine biomarkers must be greater than zero.")
    if not 18 <= age <= 120:
        raise ValueError("Phenotypic Age is only defined for adults (18-120 years).")
    v = to_equation_units(unit_system, albumin, creatinine, glucose, crp,
                          lymphocyte, mcv, rdw, alp, wbc)
    labels = {"albumin_g_l": "Albumin", "creatinine_umol_l": "Creatinine", "glucose_mmol_l": "Glucose",
              "crp_mg_dl": "CRP", "lymphocyte_pct": "Lymphocytes", "mcv_fl": "MCV",
              "rdw_pct": "RDW", "alp_u_l": "ALP", "wbc_10e9_l": "WBC"}
    for k, (lo, hi) in LIMITS.items():
        if not lo <= v[k] <= hi:
            raise ValueError(f"{labels[k]} is outside the plausible range after conversion "
                             f"({v[k]:.3g}; allowed {lo}-{hi}). Check the unit system and units.")
    notes = []
    if age < 20:
        notes.append("Age under 20: the equation was fitted on adults 20+; treat with extra caution.")
    for k, (lo, hi) in TYPICAL.items():
        if not lo <= v[k] <= hi:
            notes.append(f"{labels[k]} is outside the typical adult reference range.")
    pa, mortality, xb = phenotypic_age_core(age, **v)
    return {"phenotypic_age": round(pa, 2), "mortality_score": round(mortality, 6),
            "linear_predictor": round(xb, 4), "equation_inputs": {k: round(x, 4) for k, x in v.items()},
            "notes": notes}
