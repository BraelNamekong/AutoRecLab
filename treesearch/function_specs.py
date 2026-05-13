from treesearch.backend.utils import FunctionSpec
from dataclasses import dataclass


@dataclass
class SelectDatasets:
    """Select appropriate datasets for the recommender system research task based on the task description and available datasets."""
    selected_datasets: list[str] # A List of dataset identifiers selected for the research task.

@dataclass
class PlotAnalyses:
    """ Detailed analysis of the plot's results and implications """
    analysis: str # Detailed analysis of the plot's results and implications

@dataclass
class ReviewFunction:
    """ Submit a review evaluating the output of the training script. """
    is_bug : bool # true if the output log shows that the execution failed or has some bug, otherwise false.
    summary : str # if there is a bug, summarize the bug and propose a fix. Otherwise, leave it empty.


@dataclass
class ScoreCode:
    """Judge whether a single requirement is fulfilled by the code implementation and explain briefly."""
    fulfilled: bool # True if the specified requirement is fulfilled, false otherwise."""
    feedback: str # Short feedback explaining why the requirement is or isn't fulfilled."""

@dataclass
class ConfirmCoverage:
    """Confirm whether all requirements are fully covered by code and output."""
    confirmed: bool
    missing_requirements: list[str]
    notes: str

@dataclass
class RequirementJudgement:
    """ Judge whether a single requirement is fulfilled by the code implementation and explain briefly. """
    fulfilled : bool # True if the specified requirement is fulfilled, false otherwise.
    feedback : str # Short feedback explaining why the requirement is or isn't fulfilled.

@dataclass
class CodeRequirements:
        """Set clear and specific code requirements for the implementation based on the research task."""
        requirements: list[str]  # A list of concise, clear and specific code requirements.

@dataclass
class PlotSelection:
    """Select the 10 most relevant plots for analysis"""
    selected_plots : list[str] # description": "List of selected plot file paths

    def __post_init__(self):
        if len(self.selected_plots) >10:
            raise ValueError(" list can not exceed 10 elements")
        
@dataclass
class PlanAndCode:
    """Return a natural language plan and the Python code that implements it.
    IMPORTANT: Do not use any markdown tags or similar in the code field. It MUST be plain and executable code.
    """
    nl_text : str # Explanatory natural language text describing the plan or reasoning behind the code.
    code : str # The complete plain and executable Python source code implementing the plan.