# health_report.py

from typing import Dict, Any, List
import pandas as pd

def compute_health_metrics(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    final_df: pd.DataFrame,
    flags: List[str],
) -> Dict[str, Any]:
    """
    Calcula métricas de 'salud' del dataset antes y después de la limpieza.
    """
    report: Dict[str, Any] = {}

    n_raw, n_final = len(raw_df), len(final_df)
    report["rows_raw"] = int(n_raw)
    report["rows_final"] = int(n_final)
    report["rows_removed"] = int(n_raw - n_final)
    report["pct_removed"] = round(100 * (n_raw - n_final) / max(1, n_raw), 2)

    dup_raw = int(raw_df.duplicated().sum())
    dup_final = int(final_df.duplicated().sum())
    report["duplicates_raw"] = dup_raw
    report["duplicates_final"] = dup_final

    missing_raw = raw_df.isna().sum().to_dict()
    missing_final = final_df.isna().sum().to_dict()
    report["missing_raw"] = {k: int(v) for k, v in missing_raw.items()}
    report["missing_final"] = {k: int(v) for k, v in missing_final.items()}

    flagged_raw = 0
    flagged_final = 0
    for fc in flags:
        # En el raw puede que no existan las columnas de flags aún (se crean en limpieza)
        if fc in raw_df.columns:
            flagged_raw += int(raw_df[fc].sum())
        if fc in final_df.columns:
            flagged_final += int(final_df[fc].sum())

    report["flagged_raw"] = int(flagged_raw)
    report["flagged_final"] = int(flagged_final)

    total_missing_raw = sum(missing_raw.values())
    total_missing_final = sum(missing_final.values())
    
    # El health score considera nulos y flags como problemas
    total_issues_raw = total_missing_raw + flagged_raw
    total_issues_final = total_missing_final + flagged_final

    # Celdas totales
    cells_raw = max(1, n_raw) * max(1, raw_df.shape[1])
    cells_final = max(1, n_final) * max(1, final_df.shape[1])

    score_raw = 100 * (1 - (total_issues_raw / cells_raw))
    score_final = 100 * (1 - (total_issues_final / cells_final))

    report["health_score_raw"] = round(max(0, score_raw), 2)
    report["health_score_final"] = round(max(0, score_final), 2)

    return report
