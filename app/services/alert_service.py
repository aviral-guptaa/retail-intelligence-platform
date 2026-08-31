"""Alert and congestion-status evaluation."""
from __future__ import annotations

from typing import Dict, Tuple


class AlertService:
    """Maps live queue metrics + model predictions into NORMAL / WARNING / HIGH."""

    @staticmethod
    def assess(queue_total: int, predictions: Dict[str, float],
               alert_settings: Dict) -> Tuple[str, str]:
        warning = float(alert_settings.get("congestion_warning_queue", 4))
        high = float(alert_settings.get("congestion_high_queue", 8))
        # Only numeric horizon values participate; metadata keys (source, label)
        # are ignored so string labels cannot break the comparison.
        numeric = {k: v for k, v in predictions.items()
                   if isinstance(v, (int, float)) and k.endswith("min") and k[:-3].isdigit()}
        max_pred = max(numeric.values()) if numeric else 0.0

        if queue_total >= high or max_pred >= high:
            status = "HIGH"
            rec = (f"Queue now {queue_total}, forecast up to {max_pred:.0f} in "
                   f"{max(numeric, key=numeric.get) if numeric else '10'} minutes - open an additional counter immediately.")
        elif queue_total >= warning or max_pred >= warning:
            status = "WARNING"
            rec = f"Queue at {queue_total} (predicted {max_pred:.0f}); monitor dwell and open a counter if it grows."
        else:
            status = "NORMAL"
            rec = "No congestion expected in the near future."
        return status, rec