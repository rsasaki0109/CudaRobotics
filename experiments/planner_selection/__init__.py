from .functional_selector import FunctionalSelector
from .oop_selector import OOPSelector
from .pipeline_selector import PipelineSelector
from core.planner_selector_interface import AggregateBenchmarkRow, SelectionRequest


PROBLEM_KIND = "planner_selection"
INTERFACE_FILE = "planner_selector_interface.py"
TITLE = "Planner Selection"
DESCRIPTION_LINES = [
    "choose one planner configuration per dataset/scenario pair",
    "keep the input schema fixed while varying only the selector implementation style",
    "score each selector on benchmark regret, runtime, readability, and extensibility proxies",
]
REQUEST_SUMMARY = "dataset/scenario pairs"
METRIC_NOTES = [
    "`Avg Regret`: utility gap from an external oracle scorer; lower is better",
    "`Oracle Match`: fraction of requests where the selector picked the oracle row exactly",
    "`Readability` and `Extensibility`: static-analysis proxies, not human review replacements",
]


def build_requests(rows: list[AggregateBenchmarkRow]) -> list[SelectionRequest]:
    keys = {(row.dataset, row.scenario) for row in rows}
    return [SelectionRequest(dataset=dataset, scenario=scenario) for dataset, scenario in sorted(keys)]


def build_variants():
    return [
        FunctionalSelector(),
        OOPSelector(),
        PipelineSelector(),
    ]
