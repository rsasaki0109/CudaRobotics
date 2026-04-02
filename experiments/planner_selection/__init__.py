from .functional_selector import FunctionalSelector
from .oop_selector import OOPSelector
from .pipeline_selector import PipelineSelector


def build_variants():
    return [
        FunctionalSelector(),
        OOPSelector(),
        PipelineSelector(),
    ]
