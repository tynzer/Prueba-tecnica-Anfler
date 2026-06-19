import json
import shutil
import unittest
from pathlib import Path

from runner import OUTPUT_DIR, SummarizerExperiment


class TestRunnerSimulation(unittest.TestCase):
    def setUp(self) -> None:
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)

    def tearDown(self) -> None:
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)

    def test_simulation_runs_and_creates_outputs(self) -> None:
        experiment = SummarizerExperiment(simulate=True)
        payload = experiment.run_for_first_row()

        self.assertEqual(payload["document_executed"], 0)
        self.assertEqual(payload["executed_scope"], "first_row_only")
        self.assertEqual(len(payload["results"]), 3)
        self.assertIn("winner", payload)

        json_path = OUTPUT_DIR / "summary_experiment_results.json"
        csv_path = OUTPUT_DIR / "summary_experiment_results.csv"
        md_path = OUTPUT_DIR / "summary_experiment_report.md"

        self.assertTrue(json_path.exists())
        self.assertTrue(csv_path.exists())
        self.assertTrue(md_path.exists())

        parsed = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["executed_scope"], "first_row_only")
        self.assertEqual(len(parsed["results"]), 3)

        scores = [row["quality_score"] for row in parsed["results"]]
        self.assertTrue(all(isinstance(score, (int, float)) for score in scores))

    def test_simulation_runs_multiple_rows(self) -> None:
        experiment = SummarizerExperiment(simulate=True)
        payload = experiment.run(rows=2)

        self.assertEqual(payload["executed_rows_count"], 2)
        self.assertEqual(payload["document_indices_executed"], [0, 1])
        self.assertEqual(payload["executed_scope"], "first_2_rows")
        self.assertEqual(len(payload["documents"]), 2)

        for document in payload["documents"]:
            self.assertEqual(len(document["results"]), 3)
            self.assertIn("winner", document)


if __name__ == "__main__":
    unittest.main()
