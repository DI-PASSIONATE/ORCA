from typing import Any, Dict, Callable, Optional
from abc import ABC, abstractmethod

from orca.logger import logger


class PipelineStage(ABC):
    def __init__(self, name: str, index: int = 0):
        self.name = name
        self.index = index
        logger.debug(f"Initialized pipeline stage: {self.name}")

    @abstractmethod
    def run(
        self,
        context: Dict[str, Any],
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the pipeline stage.

        Args:
            context: A shared dictionary containing data from previous stages.
            progress_callback: A function to report progress (stage, current, total, message).

        Returns:
            Updated context dictionary.
        """
        pass
