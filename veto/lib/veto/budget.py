"""The veto budget (Guglielmo, 16 Nov 2026): the cost foreseen for the full SBT + UBT configuration, i.e. the reference
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
