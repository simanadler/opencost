"""Unit tests for the pure logic. They need no cluster: `python3 -m unittest discover -s tests`."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_it import oracle, results as R  # noqa: E402
from inference_it.metrics import delta_rows, parse, to_series  # noqa: E402
from inference_it.oracle import Reading  # noqa: E402

LITELLM = '''# HELP litellm_input_tokens_metric_total x
# TYPE litellm_input_tokens_metric_total counter
litellm_input_tokens_metric_total{api_key_alias="rag",end_user="alice",model="gemma-4-31B",team_alias="acme-corp/engineering"} 654.0
litellm_input_tokens_metric_total{api_key_alias="None",end_user="None",model="gemma-4-31B",team_alias="None"} 16.0
litellm_output_tokens_metric_total{api_key_alias="rag",end_user="alice",model="gemma-4-31B",team_alias="acme-corp/engineering"} 48.0
other_metric 3
'''

ENGINE = '''vllm:prompt_tokens_total{engine="0",model_name="gemma-4-31B"} 90.0
vllm:generation_tokens_total{engine="0",model_name="gemma-4-31B"} 44.0
vllm:prompt_tokens_cached_total{engine="0",model_name="gemma-4-31B"} 10.0
'''

SGLANG_PD = '''sglang:prompt_tokens_total{engine_type="prefill",model_name="llama"} 100.0
sglang:prompt_tokens_total{engine_type="decode",model_name="llama"} 100.0
sglang:generation_tokens_total{engine_type="decode",model_name="llama"} 40.0
'''


class ParseTests(unittest.TestCase):
    def test_parse_filters_names_and_reads_labels(self):
        s = parse(LITELLM, {"litellm_input_tokens_metric_total"})
        self.assertEqual(len(s), 2)
        rag = [x for x in s if x.label("api_key_alias") == "rag"][0]
        self.assertEqual(rag.label("team_alias"), "acme-corp/engineering")
        self.assertEqual(rag.value, 654.0)

    def test_none_sentinel_is_a_string(self):
        s = parse(LITELLM, {"litellm_input_tokens_metric_total"})
        self.assertEqual([x for x in s if x.label("team_alias") == "None"][0].value, 16.0)

    def test_slash_in_tenant_does_not_collide(self):
        text = ('litellm_input_tokens_metric_total{team_alias="acme-corp/engineering",api_key_alias="rag"} 1\n'
                'litellm_input_tokens_metric_total{team_alias="acme-corp",api_key_alias="engineering/rag"} 2\n')
        self.assertEqual(len(to_series(parse(text))), 2)

    def test_comments_and_blank_lines_ignored(self):
        self.assertEqual(parse("# TYPE x counter\n\n"), [])


class DeltaTests(unittest.TestCase):
    def test_increase_and_new_series(self):
        a = to_series(parse('m{a="1"} 10\n'))
        b = to_series(parse('m{a="1"} 15\nm{a="2"} 4\n'))
        rows = {tuple(r.labels.items()): r.delta for r in delta_rows(a, b)}
        self.assertEqual(rows, {(("a", "1"),): 5.0, (("a", "2"),): 4.0})

    def test_counter_reset_counts_new_value(self):
        a = to_series(parse('m{a="1"} 100\n'))
        b = to_series(parse('m{a="1"} 7\n'))
        self.assertEqual(delta_rows(a, b)[0].delta, 7.0)

    def test_unchanged_series_omitted(self):
        a = to_series(parse('m{a="1"} 5\n'))
        self.assertEqual(delta_rows(a, a), [])


class OracleTests(unittest.TestCase):
    def test_vllm_reading(self):
        r = oracle.reading_from_text(ENGINE)
        self.assertEqual((r.prompt, r.generation, r.cached), (90.0, 44.0, 10.0))
        self.assertEqual(r.models, frozenset({"gemma-4-31B"}))

    def test_sglang_decode_prompt_excluded(self):
        r = oracle.reading_from_text(SGLANG_PD)
        self.assertEqual(r.prompt, 100.0)
        self.assertEqual(r.generation, 40.0)

    def test_diff_sums_pods(self):
        before = {"p1": Reading(10, 5, 0, frozenset()), "p2": Reading(20, 8, 0, frozenset())}
        after = {"p1": Reading(15, 7, 0, frozenset()), "p2": Reading(30, 8, 0, frozenset())}
        d = oracle.diff(before, after)
        self.assertEqual((d.prompt, d.generation), (15.0, 2.0))
        self.assertEqual(d.warnings, [])

    def test_diff_flags_pod_restart(self):
        before = {"p1": Reading(100, 50, 0, frozenset())}
        after = {"p1": Reading(4, 2, 0, frozenset())}
        d = oracle.diff(before, after)
        self.assertEqual((d.prompt, d.generation), (4.0, 2.0))
        self.assertTrue(any("reset" in w for w in d.warnings))

    def test_diff_flags_disappeared_and_new_pods(self):
        d = oracle.diff({"gone": Reading(1, 1, 0, frozenset())}, {"new": Reading(3, 2, 0, frozenset())})
        self.assertEqual(len(d.warnings), 2)


class ResultTests(unittest.TestCase):
    def test_gap_holding_is_xfail_and_fixed_gap_fails(self):
        self.assertEqual(R.gap("G-06", "t", "s", "ref", holds=True).state, R.XFAIL)
        fixed = R.gap("G-06", "t", "s", "ref", holds=False)
        self.assertEqual(fixed.state, R.FAIL)
        self.assertIn("no longer reproduces", fixed.summary)

    def test_exit_code(self):
        self.assertEqual(R.exit_code([R.passed("a", "t"), R.gap("b", "t", "s", "r")]), 0)
        self.assertEqual(R.exit_code([R.passed("a", "t"), R.blocked("b", "t", "r")]), 1)
        self.assertEqual(R.exit_code([R.failed("a", "t", "s")]), 1)

    def test_markdown_report_has_checklist(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.md"
            R.write_markdown(path, {"run_id": "x", "versions": {"litellm": "v1"}},
                             [R.passed("P-04", "t"), R.gap("G-06", "t", "s", "ref", holds=True)])
            text = path.read_text()
        self.assertIn("Selection checklist", text)
        self.assertIn("G-06:XFAIL-GAP", text)


class RegistryTests(unittest.TestCase):
    def test_every_plan_case_is_registered_and_ordered(self):
        from inference_it import cases  # noqa: F401
        from inference_it.registry import CASES, ordered
        expected = {"P-02", "P-03", "P-04", "P-08", "P-09", "G-01", "G-02", "G-03", "G-04", "G-05",
                    "G-06", "G-07", "G-08", "G-09", "G-10", "G-11", "G-12", "G-13", "G-14", "G-15",
                    "G-16", "G-16b", "G-17"} | {f"O-{n:02d}" for n in range(1, 22)} \
            | {f"A-{n:02d}" for n in range(1, 9)}
        self.assertEqual(set(CASES), expected)
        order = [c.id for c in ordered(list(CASES.values()))]
        self.assertEqual(order[0], "P-02")
        self.assertEqual(order[-1], "A-08")
        self.assertEqual(set(order[-3:-1]), {"A-06", "A-07"})
        # phase 1 disruptive cases follow the non-disruptive phase 1 cases but precede phase 2
        self.assertLess(order.index("G-17"), order.index("G-09"))
        self.assertLess(order.index("G-16b"), order.index("O-01"))
        # the phase 2 disruptive case runs after phase 2 but before phase 3
        self.assertLess(order.index("O-21"), order.index("O-11"))
        self.assertLess(order.index("O-11"), order.index("A-01"))
        self.assertTrue(CASES["G-09"].disruptive and CASES["G-16b"].disruptive
                        and CASES["O-11"].disruptive)
        self.assertEqual(CASES["O-01"].phase, 2)
        self.assertEqual(CASES["A-01"].phase, 3)
        self.assertFalse(CASES["A-04"].disruptive)


def _entry(key, tenant="", workload="", user="", model="gemma-4-31B", ns="llm-a", cost=0.0,
           prompt=0.0, generation=0.0, in_rate=0.0, out_rate=0.0, approximate=False):
    from inference_it.opencost import Entry
    return Entry(key, model, ns, tenant, workload, user, approximate, cost, 0.0, 0.0,
                 prompt, generation, prompt + generation, in_rate, out_rate)


class Phase2PureTests(unittest.TestCase):
    """Pure logic of the Phase 2 cases: row helpers, expectations, config editing, windows."""

    def test_entry_from_json_reads_api_field_names(self):
        from inference_it.opencost import Entry
        e = Entry.from_json("gemma/llm-a/acme", {
            "properties": {"modelName": "gemma-4-31B", "namespace": "llm-a",
                           "tenant_id": "acme-corp/engineering", "workload_id": "rag",
                           "user_id": "alice", "approximate": True},
            "totalCost": 1.5, "inputCost": 1.0, "outputCost": 0.5, "promptTokens": 100.0,
            "generationTokens": 20.0, "totalTokens": 120.0,
            "inputCostPerMillionTokens": 10.0, "outputCostPerMillionTokens": 25.0})
        self.assertEqual((e.model, e.namespace, e.tenant, e.workload, e.user), (
            "gemma-4-31B", "llm-a", "acme-corp/engineering", "rag", "alice"))
        self.assertTrue(e.approximate)
        self.assertEqual((e.prompt, e.generation, e.cost), (100.0, 20.0, 1.5))

    def test_window_is_explicit_start_end(self):
        from inference_it.opencost import window
        self.assertEqual(window(1e9, 1e9 + 60), "2001-09-09T01:46:40Z,2001-09-09T01:47:40Z")

    def test_row_helpers_find_and_sum(self):
        from inference_it.cases.phase2 import find, one, tok
        rows = [_entry("a", tenant="acme", workload="rag", prompt=10, generation=2),
                _entry("b", tenant="acme", workload="batch", prompt=30, generation=4),
                _entry("c", tenant="__unmetered__", prompt=5, generation=1)]
        self.assertEqual(len(find(rows, tenant="acme")), 2)
        self.assertIsNone(one(rows, tenant="globex"))
        self.assertEqual(tok(rows, tenant="acme"), (40.0, 6.0))

    def test_matrix_expectations_sum_over_each_dimension(self):
        from inference_it.cases.phase2 import _matrix_expectations
        cells = [("t1", "w1", "u1", 10, 2), ("t1", "w1", None, 5, 1), ("t1", "w2", "u1", 7, 3),
                 ("t2", "w1", "u2", 4, 4)]
        per_tw, per_t, per_u = _matrix_expectations(cells)
        self.assertEqual(per_tw[("t1", "w1")], (15, 3))
        self.assertEqual(per_tw[("t2", "w1")], (4, 4))
        self.assertEqual(per_t["t1"], (22, 6))
        self.assertEqual(per_u["u1"], (17, 5))
        self.assertEqual(per_u[""], (5, 1))       # no-user cells key on ""

    def test_set_mapping_label_replaces_value_and_keeps_the_rest(self):
        from inference_it.lab import Lab
        text = ("producers:\n  - name: litellm-gateway\n    tenantLabel: team_alias\n"
                "    workloadLabel: api_key_alias      # hypothesis\n    userLabel: end_user\n")
        out = Lab.set_mapping_label(text, "userLabel", "")
        self.assertIn('userLabel: ""', out)
        self.assertIn("workloadLabel: api_key_alias", out)
        with self.assertRaises(ValueError):
            Lab.set_mapping_label(text, "namespaceLabel", "x")

    def test_gauge_text_parsed_with_labels(self):
        from inference_it.cases.phase2 import MODEL
        from inference_it.metrics import parse
        text = ('llm_dimension_token_coverage{direction="input",model_name="gemma-4-31B",'
                'namespace="llm-a"} 1.0\nllm_dimension_token_coverage{direction="output",'
                'model_name="gemma-4-31B",namespace="llm-a"} 0.98\n')
        samples = parse(text, {"llm_dimension_token_coverage"})
        by_direction = {s.label("direction"): s.value for s in samples}
        self.assertEqual(by_direction, {"input": 1.0, "output": 0.98})
        self.assertTrue(all(s.label("model_name") == MODEL for s in samples))

    def test_reconcile_gates_on_a_stub_lab(self):
        from inference_it.cases import phase2

        rows_by_agg = {
            phase2.MN: [_entry("m/a", ns="llm-a", cost=1.0, prompt=90, generation=18),
                        _entry("m/b", ns="llm-b", cost=2.0)],
            "model_name": [_entry("m", cost=3.0)],
            phase2.MT: [_entry("t/acme", tenant="acme", cost=1.0, prompt=90, generation=18,
                               in_rate=10.0, out_rate=25.0),
                        _entry("t/unm", tenant="__unmetered__", cost=2.0)],
        }

        class StubLab:
            def opencost(self):
                return self

            def total(self, win, basis="allocation", aggregate=None, admin=True, filter=None):
                return rows_by_agg[aggregate]

        class StubWin:
            lab = StubLab()
            t0, t1 = 0.0, 60.0
            e_in, e_out = 90.0, 18.0

            def rows(self, basis="allocation", aggregate=None, admin=True, flt=None):
                return self.lab.opencost().total(self.w(), basis=basis, aggregate=aggregate)

            def w(self):
                return "t0,t1"

        rec = phase2.reconcile(StubWin(), "allocation")
        self.assertTrue(rec["gate1"] and rec["gate2"] and rec["tokens_reconcile"])
        self.assertEqual(rec["tokens_unmetered"], (0.0, 0.0))

        # break gate 2: tenant rows over-sum the model cost
        rows_by_agg[phase2.MT][1].cost = 2.5
        rec = phase2.reconcile(StubWin(), "allocation")
        self.assertFalse(rec["gate2"])

    def test_signature_comparison_for_a03(self):
        from inference_it.cases.phase3 import _signature, _signatures_match
        rows = [_entry("m/a", tenant="acme", cost=1.0, prompt=90, generation=18),
                _entry("m/b", tenant="globex", cost=2.0, prompt=40, generation=9)]
        self.assertEqual(_signatures_match(_signature(rows), _signature(rows)), [])
        # a changed cost and a missing row are both reported
        changed = [_entry("m/a", tenant="acme", cost=1.5, prompt=90, generation=18),
                   _entry("m/b", tenant="globex", cost=2.0, prompt=40, generation=9)]
        self.assertEqual(len(_signatures_match(_signature(rows), _signature(changed))), 1)
        dropped = rows[:1]
        self.assertEqual(len(_signatures_match(_signature(rows), _signature(dropped))), 1)

    def test_hourly_cost_gauge_parsed_with_empty_user_label(self):
        from inference_it.metrics import parse
        text = ('llm_dimension_hourly_cost{cost_basis="allocation",model_name="gemma-4-31B",'
                'namespace="llm-a",tenant_id="acme-corp/engineering",workload_id="rag",'
                'user_id=""} 0.42\n'
                'llm_dimension_hourly_cost{cost_basis="allocation",model_name="gemma-4-31B",'
                'namespace="llm-a",tenant_id="acme-corp/engineering",workload_id="rag",'
                'user_id="gauge"} 0.42\n')
        samples = parse(text, {"llm_dimension_hourly_cost"})
        self.assertEqual(len(samples), 2)
        self.assertEqual({s.label("user_id") for s in samples}, {"", "gauge"})
        self.assertTrue(all(s.label("workload_id") == "rag" for s in samples))


if __name__ == "__main__":
    unittest.main()
