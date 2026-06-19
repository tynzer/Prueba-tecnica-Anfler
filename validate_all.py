import argparse
import json

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from runner import BASE_DIR, OUTPUT_DIR, SummarizerExperiment


def validate_exercise_1() -> dict:
    experiment = SummarizerExperiment(simulate=True)
    payload = experiment.run_for_first_row()

    outputs_exist = {
        "json": (OUTPUT_DIR / "summary_experiment_results.json").exists(),
        "csv": (OUTPUT_DIR / "summary_experiment_results.csv").exists(),
        "md": (OUTPUT_DIR / "summary_experiment_report.md").exists(),
    }

    return {
        "status": "ok",
        "mode": "simulated",
        "winner_model": payload["winner"]["model"],
        "winner_quality_score": payload["winner"]["quality_score"],
        "results_count": len(payload["results"]),
        "outputs_exist": outputs_exist,
        "all_outputs_present": all(outputs_exist.values()),
    }


def validate_exercise_2() -> dict:
    df = pd.read_csv(BASE_DIR / "Ejercicio 2" / "sonia_routing_dataset.csv")
    y_true = df["categoria_esperada"]
    y_pred = df["categoria_predicha"]

    labels = sorted(y_true.unique().tolist())
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "status": "ok",
        "rows": int(len(df)),
        "accuracy": round(acc, 4),
        "labels": labels,
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "confusion_matrix": cm.tolist(),
    }


def compute_approval(report: dict, min_accuracy: float, min_macro_f1: float, min_winner_quality: float) -> dict:
    checks = {
        "exercise_1_outputs": bool(report["exercise_1"]["all_outputs_present"]),
        "exercise_1_winner_quality": float(report["exercise_1"]["winner_quality_score"]) >= min_winner_quality,
        "exercise_2_accuracy": float(report["exercise_2"]["accuracy"]) >= min_accuracy,
        "exercise_2_macro_f1": float(report["exercise_2"]["macro_f1"]) >= min_macro_f1,
    }

    failed_checks = [name for name, passed in checks.items() if not passed]
    approved = len(failed_checks) == 0

    return {
        "approved": approved,
        "status": "pass" if approved else "fail",
        "thresholds": {
            "min_accuracy": min_accuracy,
            "min_macro_f1": min_macro_f1,
            "min_winner_quality": min_winner_quality,
        },
        "checks": checks,
        "failed_checks": failed_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validación integral de ambos ejercicios")
    parser.add_argument("--min-accuracy", type=float, default=0.80, help="Accuracy mínima para aprobar Ejercicio 2")
    parser.add_argument("--min-macro-f1", type=float, default=0.75, help="Macro F1 mínima para aprobar Ejercicio 2")
    parser.add_argument(
        "--min-winner-quality",
        type=float,
        default=70.0,
        help="Quality score mínimo del ganador para aprobar Ejercicio 1",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    report = {
        "exercise_1": validate_exercise_1(),
        "exercise_2": validate_exercise_2(),
    }
    report["approval"] = compute_approval(
        report,
        min_accuracy=args.min_accuracy,
        min_macro_f1=args.min_macro_f1,
        min_winner_quality=args.min_winner_quality,
    )

    out_path = OUTPUT_DIR / "validation_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Validation complete")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved report: {out_path}")

    if not report["approval"]["approved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
