from abc import ABC, abstractmethod
from typing import Callable, Optional, TYPE_CHECKING

from orca.logger import logger

if TYPE_CHECKING:
    from orca.pipeline.context import PipelineContext


class PipelineStage(ABC):
    def __init__(self, name: str, index: int = 0):
        self.name = name
        self.index = index
        logger.debug(f"Initialized pipeline stage: {self.name}")

    @abstractmethod
    def run(
        self,
        context: "PipelineContext",
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> "PipelineContext":
        """
        Execute the pipeline stage.

        Args:
            context: The current state of the run, carrying the fields earlier
                stages wrote and the folder layout of the run.
            progress_callback: A function to report progress (stage, current, total, message).

        Returns:
            The same context, updated with the fields this stage owns.
        """
