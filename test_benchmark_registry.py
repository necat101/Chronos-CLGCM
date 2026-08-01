import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

from hierarchos.evaluation import post_training
from hierarchos.evaluation import arc_agi
from hierarchos.evaluation.arc_agi import extract_json_grid, load_arc_agi_tasks
from hierarchos.evaluation.benchmarks import get_benchmark, resolve_task_names


class BenchmarkRegistryTests(unittest.TestCase):
    def test_frontier_text_expands_to_runnable_tasks_and_arc_agi_external(self):
        tasks, external, unknown = resolve_task_names(["frontier-text"], include_external=True)

        self.assertIn("mmlu_pro", tasks)
        self.assertIn("gpqa_diamond_n_shot", tasks)
        self.assertIn("aime25", tasks)
        self.assertIn("bbh_cot_fewshot", tasks)
        self.assertIn("arc_agi", {spec.key for spec in external})
        self.assertNotIn("arc_agi", tasks)
        self.assertEqual(unknown, [])

    def test_arc_agi_alias_is_first_class_external_benchmark(self):
        spec = get_benchmark("arc-agi")

        self.assertIsNotNone(spec)
        self.assertEqual(spec.key, "arc_agi")
        self.assertFalse(spec.runnable)

        tasks, external, unknown = resolve_task_names(["arc-agi"], include_external=True)
        self.assertEqual(tasks, [])
        self.assertEqual([spec.key for spec in external], ["arc_agi"])
        self.assertEqual(unknown, [])

    def test_arc_agi_family_includes_official_interactive_route(self):
        tasks, external, unknown = resolve_task_names(["arc-agi-family"], include_external=True)

        self.assertEqual(tasks, [])
        self.assertEqual([spec.key for spec in external], ["arc_agi", "arc_agi_2", "arc_agi_3"])
        self.assertEqual(unknown, [])

    def test_raw_lm_eval_tasks_are_preserved_for_new_harness_plugins(self):
        tasks, external, unknown = resolve_task_names(["new_custom_task"], include_external=True)

        self.assertEqual(tasks, ["new_custom_task"])
        self.assertEqual(external, [])
        self.assertEqual(unknown, ["new_custom_task"])

    def test_rog_ally_suite_is_bounded_to_light_runnable_tasks(self):
        tasks, external, unknown = resolve_task_names(["rog-ally"], include_external=True)

        self.assertEqual(tasks, ["arc_easy", "hellaswag", "truthfulqa_mc1"])
        self.assertEqual(external, [])
        self.assertEqual(unknown, [])

    def test_sequential_benchmark_chain_merges_scores(self):
        def fake_run_eval(**kwargs):
            task = kwargs["tasks"][0]
            return {
                "results": {task: {"acc,none": 0.5}},
                "versions": {task: "test"},
            }

        with mock.patch.object(post_training, "is_lm_eval_available", return_value=True), \
                mock.patch.object(post_training, "run_eval", side_effect=fake_run_eval) as run_eval, \
                mock.patch("builtins.print"):
            results, manifest, skipped = post_training.run_post_training_benchmarks(
                model=None,
                tokenizer=None,
                device=None,
                benchmark_names=["hellaswag", "arc_easy"],
                sequential=True,
            )

        self.assertEqual(run_eval.call_count, 2)
        self.assertEqual(set(results["results"]), {"hellaswag", "arc_easy"})
        self.assertTrue(manifest["sequential"])
        self.assertEqual(manifest["failed_lm_eval_tasks"], [])
        self.assertEqual(skipped, [])

    def test_arc_agi_json_helpers_parse_grid_and_task_file(self):
        self.assertEqual(extract_json_grid("answer: [[1,2],[3,4]]"), [[1, 2], [3, 4]])

        with tempfile.TemporaryDirectory() as tmp_dir:
            task_path = Path(tmp_dir) / "sample.json"
            task_path.write_text(
                json.dumps(
                    {
                        "train": [{"input": [[0]], "output": [[1]]}],
                        "test": [{"input": [[2]], "output": [[3]]}],
                    }
                ),
                encoding="utf-8",
            )

            tasks = load_arc_agi_tasks(str(task_path))

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_id, "sample")
        self.assertEqual(tasks[0].train[0].input, [[0]])
        self.assertEqual(tasks[0].test[0].output, [[3]])

    def test_arc_agi_uses_checkpoint_training_prompt_format(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_path = Path(tmp_dir) / "sample.json"
            task_path.write_text(
                json.dumps(
                    {
                        "train": [{"input": [[0]], "output": [[1]]}],
                        "test": [{"input": [[2]], "output": [[3]]}],
                    }
                ),
                encoding="utf-8",
            )
            model = SimpleNamespace(config=SimpleNamespace(alpaca=True))
            with mock.patch.object(
                arc_agi,
                "generate_text",
                return_value="[[3]]",
            ) as generate:
                result = arc_agi.run_arc_agi_eval(
                    model=model,
                    tokenizer=object(),
                    device=None,
                    dataset_path=str(task_path),
                )

        prompt = generate.call_args.kwargs["prompt"]
        self.assertTrue(prompt.startswith("### Instruction:\n"))
        self.assertTrue(prompt.endswith("\n\n### Response:\n"))
        self.assertEqual(
            result["config"]["prompt_format"],
            "alpaca-instruction-response",
        )


if __name__ == "__main__":
    unittest.main()
