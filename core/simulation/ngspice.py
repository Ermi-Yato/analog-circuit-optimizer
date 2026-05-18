"""
Ngspice simulator implementation.

Design:
- Each simulation runs in its own TemporaryDirectory so wrdata output files
  never collide across parallel workers.
- run_batch uses ThreadPoolExecutor: ngspice is I/O-bound in the external
  process, so threads release the GIL and give real parallelism.
- Parameter injection rule:  name -> remove underscores, uppercase, append _VAL
  e.g.  "Rc" -> "{RC_VAL}",  "R_bias1" -> "{RBIAS1_VAL}",  "C_f" -> "{CF_VAL}"
"""
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np

from core.simulation.base_simulator import BaseSimulator

_OUTPUT_FILENAME = "ngspice_simulation_output.txt"
_SIM_TIMEOUT_S = 60  # max seconds per ngspice call before we give up


class NgspiceSimulator(BaseSimulator):

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    def check_available(self) -> bool:
        return shutil.which("ngspice") is not None

    # ------------------------------------------------------------------
    # Parameter injection
    # ------------------------------------------------------------------

    @staticmethod
    def inject_params(template_str: str, params: dict) -> str:
        """
        Replace {PARAM_VAL} placeholders with numeric values.

        Naming rule: strip underscores, uppercase, wrap in {…_VAL}
            "R1"      -> "{R1_VAL}"
            "Rc"      -> "{RC_VAL}"
            "R_tail"  -> "{RTAIL_VAL}"
            "R_bias1" -> "{RBIAS1_VAL}"
            "C_f"     -> "{CF_VAL}"

        Raises:
            ValueError if a placeholder derived from any param name is not
            present in the template (catches typos early).
        """
        result = template_str
        for name, value in params.items():
            placeholder = "{" + name.replace("_", "").upper() + "_VAL}"
            if placeholder not in result:
                raise ValueError(
                    f"Placeholder {placeholder!r} not found in template. "
                    f"Check that param name {name!r} matches the template."
                )
            result = result.replace(placeholder, str(value))
        return result

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_sim_type(raw_text: str) -> str:
        """
        Detect simulation type from the column count of the first valid data row.
        AC output:         6 columns
        Transient output:  4 columns
        """
        for line in raw_text.strip().splitlines():
            parts = line.split()
            if len(parts) not in (4, 6):
                continue
            try:
                [float(p) for p in parts]
                return "ac" if len(parts) == 6 else "transient"
            except ValueError:
                continue
        return "unknown"

    @staticmethod
    def _parse_ac(raw_text: str) -> dict:
        """
        Parse AC simulation output into numpy arrays.

        ngspice wrdata column layout (1 output vector, AC analysis):
          col 0 : frequency (Hz)
          col 1-3 : frequency repeated as complex+magnitude (internal ngspice format)
          col 4 : output voltage real part
          col 5 : output voltage imaginary part

        Returns:
            {"freq": ndarray, "real": ndarray, "imag": ndarray}
            Empty dict if no valid rows found.
        """
        rows = []
        for line in raw_text.strip().splitlines():
            parts = line.split()
            if len(parts) != 6:
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue

        if not rows:
            return {}

        data = np.array(rows)
        return {
            "freq": data[:, 0],
            "real": data[:, 4],
            "imag": data[:, 5],
        }

    @staticmethod
    def _parse_transient(raw_text: str) -> dict:
        """
        Parse transient simulation output into numpy arrays.

        ngspice wrdata column layout (1 output vector, transient analysis):
          col 0 : time (s)
          col 1-2 : time repeated (internal ngspice format)
          col 3 : output voltage (V)

        Returns:
            {"time": ndarray, "voltage": ndarray}
            Empty dict if no valid rows found.
        """
        rows = []
        for line in raw_text.strip().splitlines():
            parts = line.split()
            if len(parts) != 4:
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue

        if not rows:
            return {}

        data = np.array(rows)
        return {
            "time": data[:, 0],
            "voltage": data[:, 3],
        }

    # ------------------------------------------------------------------
    # Single run (internal)
    # ------------------------------------------------------------------

    def _run_one(self, template_str: str, params: dict) -> dict | None:
        """
        Inject params, run ngspice in an isolated temp directory, parse output.

        Each call gets its own TemporaryDirectory so parallel workers never
        collide on the wrdata output file.

        Returns parsed data dict, or None on any failure.
        """
        try:
            netlist = self.inject_params(template_str, params)
        except ValueError:
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            netlist_path = os.path.join(tmpdir, "sim.cir")
            output_path  = os.path.join(tmpdir, _OUTPUT_FILENAME)

            with open(netlist_path, "w", encoding="utf-8") as f:
                f.write(netlist)

            try:
                subprocess.run(
                    ["ngspice", "-b", netlist_path],
                    capture_output=True,
                    cwd=tmpdir,
                    timeout=_SIM_TIMEOUT_S,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return None

            if not os.path.exists(output_path):
                return None

            with open(output_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

        # raw_text is in memory — temp dir can now be deleted safely
        sim_type = self._detect_sim_type(raw_text)
        if sim_type == "ac":
            return self._parse_ac(raw_text) or None
        if sim_type == "transient":
            return self._parse_transient(raw_text) or None
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_single(self, template_str: str, params: dict) -> dict | None:
        return self._run_one(template_str, params)

    def run_batch(
        self,
        template_str: str,
        param_list: list[dict],
        progress_callback: Callable[[int, int], None] | None = None,
        max_workers: int | None = None,
    ) -> list[dict | None]:
        """
        Run all simulations in parallel via ThreadPoolExecutor.

        Results are returned in the same order as param_list regardless of
        which worker finishes first.
        """
        total = len(param_list)
        results: list[dict | None] = [None] * total
        completed_count = 0
        lock = threading.Lock()

        def _worker(idx: int, params: dict):
            nonlocal completed_count
            data = self._run_one(template_str, params)
            with lock:
                completed_count += 1
                count = completed_count
            if progress_callback is not None:
                progress_callback(count, total)
            return idx, data

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_worker, i, p): i
                for i, p in enumerate(param_list)
            }
            for future in as_completed(futures):
                idx, data = future.result()
                results[idx] = data

        return results
