import argparse
import json
import math
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
EJ1_DIR = BASE_DIR / "Ejercicio 1"
DATASET_PATH = EJ1_DIR / "dataset_resumidor_muestra11.csv"
PROMPT_PATH = EJ1_DIR / "prompt_resumidor.md"
OUTPUT_DIR = BASE_DIR / "outputs"

JUDGE_MODEL = "gpt-5.4"
CANDIDATE_MODELS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
]


@dataclass
class SummaryRun:
    document_index: int
    model: str
    summary: str
    quality_score: float
    coverage_score: float
    factuality_score: float
    clarity_score: float
    concision_score: float
    strengths: List[str]
    weaknesses: List[str]
    missing_critical_points: List[str]
    token_usage: Dict[str, Any]
    latency_seconds: float


class SummarizerExperiment:
    def __init__(self, simulate: Optional[bool] = None) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        self.simulate = (not bool(api_key)) if simulate is None else simulate
        self.prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

        if self.simulate:
            self.client = None
            return

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY no está configurada. Creá un archivo .env o exportá la variable antes de ejecutar."
            )
        self.client = OpenAI(api_key=api_key)

    def load_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(DATASET_PATH)
        if df.empty:
            raise ValueError("El dataset de resumidor está vacío.")
        return df

    def build_summary_prompt(self, text: str) -> str:
        return self.prompt_template.replace("{texto_original}", text)

    def generate_summary(self, model: str, text: str) -> Dict[str, Any]:
        import time

        if self.simulate:
            return self.simulate_summary(model, text)

        started = time.perf_counter()
        response = self.client.responses.create(
            model=model,
            input=self.build_summary_prompt(text),
        )
        latency = time.perf_counter() - started
        return {
            "summary": response.output_text.strip(),
            "usage": getattr(response, "usage", None),
            "latency_seconds": round(latency, 3),
        }

    def judge_summary(self, text: str, critical_points: str, summary: str) -> Dict[str, Any]:
        if self.simulate:
            return self.simulate_judge(text, critical_points, summary)

        judge_prompt = f"""
Sos un evaluador experto de resúmenes documentales.

Tu tarea es puntuar la calidad del resumen usando como referencia:
1. el documento original,
2. los puntos críticos esperados,
3. el resumen generado.

Devolvé exclusivamente un JSON válido con esta estructura exacta:
{{
  "quality_score": 0-100,
  "coverage_score": 0-100,
  "factuality_score": 0-100,
  "clarity_score": 0-100,
  "concision_score": 0-100,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "missing_critical_points": ["..."]
}}

Criterios:
- quality_score: evaluación global.
- coverage_score: qué tan bien cubre puntos críticos.
- factuality_score: fidelidad al documento, sin invenciones.
- clarity_score: organización, legibilidad y utilidad.
- concision_score: nivel de síntesis sin sacrificar información relevante.
- strengths/weaknesses: máximo 3 ítems por lista.
- missing_critical_points: listar solo ausencias relevantes.

Documento original:
{text}

Puntos críticos esperados:
{critical_points}

Resumen generado:
{summary}
"""
        response = self.client.responses.create(
            model=JUDGE_MODEL,
            input=judge_prompt,
        )
        raw = response.output_text.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"No se pudo parsear la respuesta del juez: {raw}")
        result = json.loads(match.group(0))
        result["usage"] = getattr(response, "usage", None)
        return result

    @staticmethod
    def _first_sentences(text: str, n: int) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        selected = [s for s in sentences if s][:n]
        return " ".join(selected).strip()

    def simulate_summary(self, model: str, text: str) -> Dict[str, Any]:
        base = self._first_sentences(text, 4)
        style_suffix = {
            "gpt-4.1-mini": " Prioriza síntesis breve y operativa.",
            "gpt-4.1": " Incluye detalle equilibrado entre contexto, impacto y acciones.",
            "gpt-4o-mini": " Enfatiza claridad y próximos pasos accionables.",
        }.get(model, "")

        summary = f"{base}{style_suffix}".strip()
        return {
            "summary": summary,
            "usage": {
                "input_tokens": 0,
                "output_tokens": max(1, len(summary.split())),
                "total_tokens": max(1, len(summary.split())),
                "mode": "simulated",
            },
            "latency_seconds": 0.01,
        }

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        words = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúñÑ0-9]+", text.lower())
        stopwords = {
            "de", "la", "el", "y", "en", "que", "los", "las", "un", "una", "por", "con",
            "del", "al", "se", "para", "su", "más", "sin", "como", "entre", "fue", "antes",
            "sobre", "a", "o", "lo", "le", "es", "si", "no", "ya", "sus", "esta", "este",
        }
        return [w for w in words if len(w) > 3 and w not in stopwords]

    def simulate_judge(self, text: str, critical_points: str, summary: str) -> Dict[str, Any]:
        critical_keywords = set(self._extract_keywords(critical_points))
        summary_keywords = set(self._extract_keywords(summary))

        if not critical_keywords:
            overlap_ratio = 0.7
        else:
            overlap_ratio = len(critical_keywords & summary_keywords) / len(critical_keywords)

        coverage = round(55 + overlap_ratio * 40, 1)
        factuality = 88.0
        clarity = 82.0 if len(summary.split()) > 40 else 76.0
        concision = 84.0 if len(summary.split()) < 140 else 72.0
        quality = round((coverage * 0.45) + (factuality * 0.25) + (clarity * 0.2) + (concision * 0.1), 1)

        missing_points = []
        if overlap_ratio < 0.5:
            missing_points.append("Falta cubrir una parte importante de los puntos críticos esperados")

        return {
            "quality_score": quality,
            "coverage_score": coverage,
            "factuality_score": factuality,
            "clarity_score": clarity,
            "concision_score": concision,
            "strengths": [
                "Resumen coherente y legible",
                "Mantiene foco en el problema operativo",
            ],
            "weaknesses": [
                "Evaluación simulada sin juez externo",
            ],
            "missing_critical_points": missing_points,
            "usage": {"mode": "simulated", "total_tokens": 0},
        }

    @staticmethod
    def normalize_usage(usage: Any) -> Dict[str, Any]:
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if isinstance(usage, dict):
            return usage
        return {"raw": str(usage)}

    def evaluate_document(self, document_index: int, text: str, critical_points: str) -> Dict[str, Any]:
        runs: List[SummaryRun] = []
        for model in CANDIDATE_MODELS:
            summary_result = self.generate_summary(model, text)
            judge_result = self.judge_summary(text, critical_points, summary_result["summary"])
            runs.append(
                SummaryRun(
                    document_index=document_index,
                    model=model,
                    summary=summary_result["summary"],
                    quality_score=float(judge_result["quality_score"]),
                    coverage_score=float(judge_result["coverage_score"]),
                    factuality_score=float(judge_result["factuality_score"]),
                    clarity_score=float(judge_result["clarity_score"]),
                    concision_score=float(judge_result["concision_score"]),
                    strengths=judge_result.get("strengths", []),
                    weaknesses=judge_result.get("weaknesses", []),
                    missing_critical_points=judge_result.get("missing_critical_points", []),
                    token_usage={
                        "summarizer": self.normalize_usage(summary_result["usage"]),
                        "judge": self.normalize_usage(judge_result.get("usage")),
                    },
                    latency_seconds=summary_result["latency_seconds"],
                )
            )

        results_df = pd.DataFrame([asdict(run) for run in runs]).sort_values(
            by=["quality_score", "coverage_score", "factuality_score", "clarity_score"],
            ascending=False,
        )
        winner = results_df.iloc[0].to_dict()
        return {
            "document_index": document_index,
            "winner": winner,
            "results": results_df.to_dict(orient="records"),
        }

    def run(self, rows: Optional[int] = 1) -> Dict[str, Any]:
        OUTPUT_DIR.mkdir(exist_ok=True)
        df = self.load_dataset()

        if rows is not None and rows <= 0:
            raise ValueError("rows debe ser mayor a 0 o None para ejecutar todas las filas")

        selected_df = df if rows is None else df.head(rows)
        documents = []
        for document_index, row in selected_df.iterrows():
            documents.append(
                self.evaluate_document(
                    document_index=int(document_index),
                    text=row["texto_original"],
                    critical_points=row["puntos_criticos"],
                )
            )

        flattened_results: List[Dict[str, Any]] = []
        for document in documents:
            flattened_results.extend(document["results"])

        results_df = pd.DataFrame(flattened_results)
        json_path = OUTPUT_DIR / "summary_experiment_results.json"
        csv_path = OUTPUT_DIR / "summary_experiment_results.csv"
        md_path = OUTPUT_DIR / "summary_experiment_report.md"

        executed_scope = "all_rows" if rows is None else ("first_row_only" if rows == 1 else f"first_{rows}_rows")
        payload = {
            "executed_rows_count": int(len(selected_df)),
            "document_indices_executed": [int(i) for i in selected_df.index.tolist()],
            "total_rows_available": int(len(df)),
            "executed_scope": executed_scope,
            "judge_model": JUDGE_MODEL,
            "candidate_models": CANDIDATE_MODELS,
            "documents": documents,
            # Mantiene compatibilidad con consumidores existentes para ejecución de una fila.
            "document_executed": int(selected_df.index.tolist()[0]) if len(selected_df) == 1 else None,
            "winner": documents[0]["winner"] if len(documents) == 1 else None,
            "results": documents[0]["results"] if len(documents) == 1 else [],
        }

        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        results_df.to_csv(csv_path, index=False)
        md_path.write_text(self.build_markdown_report(payload), encoding="utf-8")
        return payload

    def run_for_first_row(self) -> Dict[str, Any]:
        return self.run(rows=1)

    @staticmethod
    def build_markdown_report(payload: Dict[str, Any]) -> str:
        document_reports = payload.get("documents", [])
        first_winner = payload.get("winner") or (document_reports[0]["winner"] if document_reports else {})
        lines = [
            "# Reporte del experimento de resumidor",
            "",
            f"- Cantidad de documentos ejecutados: {payload.get('executed_rows_count', 0)}",
            f"- Índices ejecutados: {payload.get('document_indices_executed', [])}",
            f"- Cantidad total de filas disponibles: {payload['total_rows_available']}",
            f"- Alcance ejecutado: {payload['executed_scope']}",
            f"- Modelo juez: {payload['judge_model']}",
            f"- Modelo ganador (primer documento): {first_winner.get('model', 'N/A')}",
            f"- Puntaje global (primer documento): {first_winner.get('quality_score', 'N/A')}",
            "",
            "## Resultados por modelo",
            "",
        ]
        for document in document_reports:
            lines.append(f"### Documento índice {document['document_index']}")
            lines.append("")
            for row in document["results"]:
                lines.extend(
                    [
                        f"#### {row['model']}",
                        f"- quality_score: {row['quality_score']}",
                        f"- coverage_score: {row['coverage_score']}",
                        f"- factuality_score: {row['factuality_score']}",
                        f"- clarity_score: {row['clarity_score']}",
                        f"- concision_score: {row['concision_score']}",
                        f"- latency_seconds: {row['latency_seconds']}",
                        f"- strengths: {', '.join(row['strengths']) if row['strengths'] else 'N/A'}",
                        f"- weaknesses: {', '.join(row['weaknesses']) if row['weaknesses'] else 'N/A'}",
                        f"- missing_critical_points: {', '.join(row['missing_critical_points']) if row['missing_critical_points'] else 'Ninguno'}",
                        "",
                        "##### Resumen generado",
                        row["summary"],
                        "",
                    ]
                )
        return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta el experimento del componente resumidor")
    parser.add_argument(
        "--mode",
        choices=["auto", "simulate", "real"],
        default="auto",
        help="Modo de ejecución: auto (usa API key si existe), simulate o real",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Alias legacy de --mode simulate",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1,
        help="Cantidad de filas a ejecutar desde el inicio del dataset (default: 1)",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Ejecuta todas las filas del dataset (escalable a 1000)",
    )
    args = parser.parse_args()

    selected_mode = "simulate" if args.simulate else args.mode
    if selected_mode == "auto":
        experiment = SummarizerExperiment(simulate=None)
    elif selected_mode == "simulate":
        experiment = SummarizerExperiment(simulate=True)
    else:
        experiment = SummarizerExperiment(simulate=False)

    rows_to_run = None if args.all_rows else args.rows
    result = experiment.run(rows=rows_to_run)
    print(f"execution_mode={ 'simulated' if experiment.simulate else 'real' }")
    print(f"executed_rows_count={result['executed_rows_count']}")
    if result.get("winner"):
        print(json.dumps(result["winner"], ensure_ascii=False, indent=2))
