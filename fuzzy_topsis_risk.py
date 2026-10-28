"""Fuzzy TOPSIS for supply chain risk prioritization.

Edit RISKS, CRITERIA, and the rating dictionaries, then run:
    python fuzzy_topsis_risk.py
Requires only numpy.
"""
import numpy as np

# Linguistic scale -> triangular fuzzy numbers (l, m, u)
RATING = {
    "VL": (0.0, 0.0, 0.25),
    "L":  (0.0, 0.25, 0.5),
    "M":  (0.25, 0.5, 0.75),
    "H":  (0.5, 0.75, 1.0),
    "VH": (0.75, 1.0, 1.0),
}

RISKS = ["Supplier failure", "Port congestion", "Demand volatility",
         "Cyberattack", "Natural disaster", "Machine breakdown"]
CRITERIA = ["Probability", "Severity", "Detectability", "Recovery time"]
# All criteria are "higher = riskier" (detectability = difficulty of detection)

# Expert ratings: ratings[expert][risk] = one term per criterion
RATINGS = [
    [["M", "VH", "M", "H"], ["H", "M", "L", "M"], ["H", "H", "M", "M"],
     ["L", "VH", "H", "H"], ["VL", "VH", "M", "VH"], ["H", "M", "L", "L"]],
    [["M", "H", "H", "H"], ["VH", "M", "M", "M"], ["H", "M", "H", "L"],
     ["M", "H", "VH", "H"], ["L", "VH", "L", "VH"], ["M", "H", "L", "M"]],
    [["L", "VH", "M", "VH"], ["H", "H", "L", "M"], ["VH", "H", "M", "M"],
     ["L", "H", "H", "M"], ["VL", "H", "M", "H"], ["H", "M", "M", "L"]],
]
# Criteria importance per expert (same linguistic scale)
WEIGHTS = [["M", "VH", "H", "H"], ["H", "VH", "M", "H"], ["M", "H", "H", "VH"]]


def aggregate(terms):
    """Average the fuzzy numbers across experts: l=mean, m=mean, u=mean."""
    arr = np.array([[RATING[t] for t in row] for row in terms])
    return arr.mean(axis=0)


def fuzzy_topsis(matrix, weights):
    """matrix: (alts, crit, 3); weights: (crit, 3). All criteria benefit-type."""
    cstar = matrix[:, :, 2].max(axis=0)                  # max upper value per criterion
    norm = matrix / cstar[None, :, None]                 # normalized fuzzy matrix
    v = norm * weights[None, :, :]                       # weighted fuzzy matrix
    d_plus = np.sqrt(((v - 1.0) ** 2).mean(axis=2)).sum(axis=1)   # vertex distance to FPIS
    d_minus = np.sqrt((v ** 2).mean(axis=2)).sum(axis=1)          # to FNIS
    return d_minus / (d_plus + d_minus), d_plus, d_minus


def rank(scores):
    order = np.argsort(-scores)
    r = np.empty(len(scores), dtype=int)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def spearman(r1, r2):
    n = len(r1)
    return 1 - 6 * np.sum((r1 - r2) ** 2) / (n * (n ** 2 - 1))


def run(weights_fuzzy, label):
    scores, _, _ = fuzzy_topsis(MATRIX, weights_fuzzy)
    return scores


# --- Build aggregated decision matrix (risks x criteria x 3) ---
n_exp = len(RATINGS)
MATRIX = np.zeros((len(RISKS), len(CRITERIA), 3))
for i in range(len(RISKS)):
    for j in range(len(CRITERIA)):
        MATRIX[i, j] = np.array([RATING[RATINGS[e][i][j]] for e in range(n_exp)]).mean(axis=0)
W = aggregate(WEIGHTS)  # (crit, 3)

# --- Fuzzy TOPSIS ranking ---
cc, dp, dm = fuzzy_topsis(MATRIX, W)
topsis_rank = rank(cc)

# --- Traditional RPN-style comparison (defuzzified mean, product of first 3 criteria) ---
crisp = MATRIX.mean(axis=2)
rpn = crisp[:, 0] * crisp[:, 1] * crisp[:, 2]
rpn_rank = rank(rpn)

print(f"{'Risk':<20}{'d+':>7}{'d-':>7}{'CC':>8}{'TOPSIS':>8}{'RPN':>6}")
for i in np.argsort(-cc):
    print(f"{RISKS[i]:<20}{dp[i]:7.3f}{dm[i]:7.3f}{cc[i]:8.3f}{topsis_rank[i]:8d}{rpn_rank[i]:6d}")
print(f"\nSpearman (TOPSIS vs RPN): {spearman(topsis_rank, rpn_rank):.3f}")

# --- Sensitivity analysis: scale each criterion weight by 0.8 and 1.2 ---
print("\nSensitivity (rank of each risk when one criterion weight changes):")
for j, c in enumerate(CRITERIA):
    for f in (0.8, 1.2):
        Wm = W.copy()
        Wm[j] *= f
        s, _, _ = fuzzy_topsis(MATRIX, Wm)
        r = rank(s)
        changed = "stable" if np.array_equal(r, topsis_rank) else "RANK CHANGE"
        print(f"  {c} x{f}: {[int(x) for x in r]}  {changed}")
