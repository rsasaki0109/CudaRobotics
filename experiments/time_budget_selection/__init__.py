from .functional_budget_selector import FunctionalBudgetSelector
from .oop_budget_selector import OOPBudgetSelector
from .pipeline_budget_selector import PipelineBudgetSelector


def build_variants():
    return [
        FunctionalBudgetSelector(),
        OOPBudgetSelector(),
        PipelineBudgetSelector(),
    ]
