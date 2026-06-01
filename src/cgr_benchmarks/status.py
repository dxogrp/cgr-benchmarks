from __future__ import annotations

import pandas as pd


def is_success_status(status: object) -> bool:
    if status is None or pd.isna(status):
        return False
    text = str(status).strip()
    lowered = text.lower()
    if lowered in {"optimal", "optimal_inaccurate", "solved", "almostsolved", "almost solved"}:
        return True
    code = text.split(maxsplit=1)[0]
    return code in {"1", "4"}


def status_error(status: object) -> str | None:
    return None if is_success_status(status) else f"non-success status: {status}"


def successful_solve_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame
    if "error" in rows.columns:
        rows = rows[rows["error"].isna()]
    if "status" in rows.columns:
        rows = rows[rows["status"].map(is_success_status)]
    return rows
