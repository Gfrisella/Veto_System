"""Cost unit and budget of the veto (Guglielmo, 16 Nov 2026: "instead of speaking of money let's use a token").

Token: the DRACHMA. 1 drachma = the cost of the cheapest station a design can add or remove, one module of 4 x 4 cm
scintillating tiles (0.5 x 0.5 m, 144 SiPM channels) in the placeholder cost model. kCHF stay inside the cost model only;
everything shown (reports, the optimiser's prompt, the report) is in drachmas. Budget = R2 in drachmas (333.5 on 16 Nov).
(Cheaper unit not taken: one SBT cell, 0.27 drachma, but the SBT is added a ring at a time, ~16 drachmas per ring.)

The veto budget (Guglielmo, 16 Nov 2026): the cost foreseen for the full SBT + UBT configuration, i.e. the reference
design R2 (designs/R2_sbt_ubt.json) priced with the same placeholder cost model as every other design. Costs are quoted
relative to it (R2 = 100 %). Computed, not hard-coded, so it follows changes of the cost model.

The 2023 cost table (SBT ~59 %, UBT ~2.5 % of an 8 MCHF reference) only fixes the SBT price (technology.REF_BUDGET_KCHF);
its UBT was the earlier small one, so 61.5 % of 8 MCHF would put R2 itself 37 % over budget.
"""
from functools import lru_cache

REFERENCE_DESIGN = "R2_sbt_ubt"


@lru_cache(maxsize=1)
def kchf():
    from . import design
    return float(design.load(REFERENCE_DESIGN).cost)


def pct(cost_kchf):
    """Cost in % of the budget (R2 = 100 %)."""
    return 100.0 * cost_kchf / kchf()


DRACHMA_UNIT = "one module of 4 x 4 cm scintillating tiles (0.5 x 0.5 m)"


@lru_cache(maxsize=1)
def drachma_kchf():
    """kCHF of one drachma (internal; the cost model's price of a tile module)."""
    from . import technology
    return float(technology.module_cost("tile"))


def drachma(cost_kchf):
    """A cost [kCHF of the cost model] in drachmas."""
    return cost_kchf / drachma_kchf()


def budget_drachma():
    return drachma(kchf())
