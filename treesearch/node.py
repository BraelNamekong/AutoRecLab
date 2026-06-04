import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Self

from anytree import NodeMixin

from treesearch.interpreter import ExecutionResult
from treesearch.metric import MetricValue
from treesearch.type_checker import TypeCheckResult
from treesearch.utils.response import trim_long_string


@dataclass
class Requirement:
    description: str
    is_fulfilled = False
    feedback: Optional[str] = None


@dataclass
class NodeScore:
    score: float = 0.0
    feedback: str = ""
    is_satisfactory: bool = False


@dataclass(eq=False, kw_only=True)
class Node(NodeMixin):
    """A single node in the solution tree. Contains code, execution results, and evaluation information."""

    # ---- code & plan ----
    plan: str = field(default="")  # type: ignore
    code: str = field(default="")  # type: ignore

    # ---- general attr ----
    _parent: Optional[Self] = field(default=None)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ctime: float = field(default_factory=lambda: time.time())
    exp_results_dir: str = field(default=None)  # type: ignore
    score: NodeScore = field(default_factory=NodeScore)

    # ---- execution info ----
    _term_out: list[str] = field(default=None)  # type: ignore
    exec_time: float = field(default=None)  # type: ignore

    # ---- evaluation ----
    # post-execution result analysis (findings/feedback)
    analysis: str = field(default=None)  # type: ignore
    metric: MetricValue = field(default=None)  # type: ignore
    # whether the agent decided that the code is buggy
    # -> always True if exc_type is not None or no valid metric
    is_buggy: bool = field(default=None)  # type: ignore
    requirements: list[Requirement] = field(default_factory=list)

    # ---- execution time feedback ----
    exec_time_feedback: str = field(default="")

    # ---- type checking info ----
    type_check_attempts: int = field(default=0)
    type_check_passed: bool = field(default=False)
    type_check_results: list[TypeCheckResult] = field(default_factory=list)

    # ---- debug retry limiting ----
    debug_attempts: int = field(default=0)

    @property
    def name(self) -> str:
        short_id = f"{self.id[:4]}...{self.id[-4:]}"
        if len(self.plan) > 0:
            plan_max_chars = 25
            dots = "..." if len(self.plan) > plan_max_chars else ""
            return f"{__class__.__name__}({short_id},\n{self.plan[:plan_max_chars]}{dots}\nbuggy={self.is_buggy}\nscore={self.score.score})"
        else:
            return f"{__class__.__name__}({short_id}\nbuggy={self.is_buggy}\nscore={self.score.score})"

    def __post_init__(self):
        self.parent = self._parent

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        # Create a new instance with copied attributes
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy all attributes except parent and children to avoid circular references
        for k, v in self.__dict__.items():
            if k not in ("parent", "children"):
                setattr(result, k, copy.deepcopy(v, memo))

        # Handle parent and children separately
        result.parent = self.parent  # Keep the same parent reference
        result.children = set()  # Start with empty children set

        return result

    def __getstate__(self):
        """Return state for pickling"""
        state = self.__dict__.copy()
        # Ensure id is included in the state
        if hasattr(self, "id"):
            state["id"] = self.id
        return state

    def __setstate__(self, state):
        """Set state during unpickling"""
        # Ensure all required attributes are present
        self.__dict__.update(state)
        if "debug_attempts" not in self.__dict__:
            self.debug_attempts = 0

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """Absorb the result of executing the code from this node."""
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time

    @property
    def term_out(self) -> str:
        """Get the terminal output of the code execution (after truncating it)."""
        return trim_long_string("".join(self._term_out))
