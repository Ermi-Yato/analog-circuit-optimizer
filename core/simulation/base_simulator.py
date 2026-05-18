"""Abstract base class for SPICE simulators."""
from abc import ABC, abstractmethod
from typing import Callable


class BaseSimulator(ABC):

    @abstractmethod
    def check_available(self) -> bool:
        """Return True if the simulator executable is on PATH."""
        ...

    @abstractmethod
    def run_single(self, template_str: str, params: dict) -> dict | None:
        """
        Run one simulation and return parsed data arrays.

        Args:
            template_str: SPICE template with {PARAM_VAL} placeholders.
            params: mapping of param name -> numeric value.
                    e.g. {"R1": 10000, "Rc": 3000}

        Returns:
            AC sim:         {"freq": ndarray, "real": ndarray, "imag": ndarray}
            Transient sim:  {"time": ndarray, "voltage": ndarray}
            None if the simulation failed or produced no output.
        """
        ...

    @abstractmethod
    def run_batch(
        self,
        template_str: str,
        param_list: list[dict],
        progress_callback: Callable[[int, int], None] | None = None,
        max_workers: int | None = None,
    ) -> list[dict | None]:
        """
        Run N simulations in parallel.

        Args:
            template_str:      SPICE template shared by all runs.
            param_list:        List of N param dicts, one per simulation.
            progress_callback: Called as (completed: int, total: int) after
                               each simulation completes. Safe to use for
                               GUI progress bars (called from worker threads).
            max_workers:       Worker thread count. Defaults to ThreadPoolExecutor
                               default (min(32, cpu_count + 4)).

        Returns:
            List of N result dicts preserving the input order.
            Entries are None for simulations that failed.
        """
        ...
