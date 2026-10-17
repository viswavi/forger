"""Forger's TinkerModel shares the agent's Tinker ledger and evaluator rendering."""

import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding.rl.tinker_budget import TinkerBudget, TinkerBudgetExceeded

from tests.test_agent_token_budget import launcher_functions

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
QWEN = "Qwen/Qwen3-8B"


def forger_ledger_module():
    """Load pyforger/tinker_usage.py from this checkout (standard library only)."""
    spec = importlib.util.spec_from_file_location(
        "forger_tinker_usage", REPO / "src/pyforger/tinker_usage.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LedgerCompatibilityTests(unittest.TestCase):
    """Both ledger implementations read and charge the same file."""

    def test_forger_and_benchmark_charges_share_one_cap(self):
        """Spend from either side counts against the other's checks."""
        usage = forger_ledger_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tinker_sft_usage.json"
            benchmark = TinkerBudget(path, 1.0, name="sft")
            benchmark.snapshot()  # the launcher creates the ledger
            forger = usage.TinkerUsageLedger(path, 1.0, model=QWEN)
            billed = usage.sample_tokens(1000, 4, 200, [500, 400, 300, 200])
            forger.record(billed, "forger-session")
            state = benchmark.check()
            self.assertEqual(state["calls"], 1)
            self.assertEqual(state["budget"], "sft")
            self.assertEqual(state["sessions"], ["forger-session"])
            self.assertAlmostEqual(
                state["cost_usd"], usage.tinker_cost_usd(billed, forger.prices)
            )
            # The two sides price sample requests identically.
            from coding.rl.tinker_budget import cost_usd, sample_tokens

            self.assertEqual(billed, sample_tokens(1000, 4, 200, [500, 400, 300, 200]))
            self.assertAlmostEqual(forger.cost(billed), cost_usd(QWEN, billed))
            benchmark.record({"train": 2_000_000})  # $0.88: an SFT trial
            with self.assertRaisesRegex(usage.TinkerBudgetExceeded, "sft budget"):
                forger.check(0.2)
            with self.assertRaises(TinkerBudgetExceeded):
                benchmark.check(0.2)
            self.assertEqual(json.loads(path.read_text())["calls"], 2)

    def test_rate_cards_match(self):
        """The built-in pyforger prices are the benchmark's prices."""
        from coding.rl.tinker_budget import PRICES_PER_MILLION

        usage = forger_ledger_module()
        self.assertEqual(usage.TINKER_PRICES_PER_MILLION, PRICES_PER_MILLION)
        self.assertEqual(usage.DEFAULT_MAX_CONCURRENT_SAMPLES, 600)


class LauncherWiringTests(unittest.TestCase):
    """The sandbox charges TinkerModel to the SFT ledger and pins Tinker evaluation."""

    def test_forger_tinker_ledger_is_the_sft_ledger(self):
        """configure_run_token_budgets points Forger at the SFT ledger and limit."""
        configure = launcher_functions("configure_run_token_budgets")[
            "configure_run_token_budgets"
        ]
        usage = forger_ledger_module()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {}, clear=True
        ):
            configure(directory, "coding")
            self.assertEqual(
                os.environ["FORGER_TINKER_USAGE_FILE"],
                os.environ["TINKER_SFT_USAGE_FILE"],
            )
            self.assertEqual(
                os.environ["FORGER_TINKER_USD_LIMIT"],
                os.environ["TINKER_SFT_USD_LIMIT"],
            )
            path, limit = usage.TinkerUsageLedger.env_config()
            # Opening the launcher's ledger succeeds: same model and limit.
            usage.TinkerUsageLedger(path, limit, model=QWEN).check()

    def test_sandbox_receives_forger_ledger_and_backend(self):
        """The workspace forwards the ledger variables and the evaluation backend."""
        tree = ast.parse((ROOT / "launch_agent.py").read_text())
        forwarded = next(
            ast.literal_eval(keyword.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "forward_env"
        )
        for name in (
            "FORGER_TINKER_USAGE_FILE",
            "FORGER_TINKER_USD_LIMIT",
            "PDB_EVAL_BACKEND",
            "PDB_STUDENT_MODEL",
            "TINKER_EVAL_BUDGET",
        ):
            self.assertIn(name, forwarded)
        source = (ROOT / "launch_agent.py").read_text()
        self.assertIn('os.environ["PDB_EVAL_BACKEND"] = "tinker"', source)
        # eval_launcher.sh defaults --model-path to the student being trained.
        self.assertIn('os.environ["PDB_STUDENT_MODEL"] = args.model', source)
        # Saved before the session and restored after it, so host-side
        # secret-test evaluation keeps its own backend.
        restored = next(
            {
                constant.value
                for constant in ast.walk(node.value)
                if isinstance(constant, ast.Constant)
            }
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_HOST_RL_CACHE_ENV"
                for target in node.targets
            )
        )
        self.assertIn("PDB_EVAL_BACKEND", restored)
        self.assertIn("PDB_STUDENT_MODEL", restored)


class EvalLauncherBackendTests(unittest.TestCase):
    """Tinker is the default backend; PDB_EVAL_BACKEND makes it the only one."""

    def run_launcher(
        self,
        *args,
        backend="tinker",
        student=None,
        launcher="coding/eval_launcher.sh",
        **environ,
    ):
        """Run a launcher with a fake python that logs its arguments.

        `launcher` is eval_launcher.sh (given a scratch task directory) or a
        task's evaluate_model.sh. Returns (result, logged python calls, the
        scratch root).
        """
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        bin_dir, task = root / "bin", root / "task"
        bin_dir.mkdir()
        task.mkdir()
        log = root / "python.log"
        fake = bin_dir / "python"
        fake.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{log}"\n')
        fake.chmod(0o755)
        # No GPUs: a local-server run stops before starting any server.
        no_gpu = bin_dir / "nvidia-smi"
        no_gpu.write_text("#!/usr/bin/env bash\nexit 1\n")
        no_gpu.chmod(0o755)
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PDB_EVAL_BACKEND": backend,
        }
        for name in ("SERVER_BACKEND", "PDB_STUDENT_MODEL", "CUDA_VISIBLE_DEVICES"):
            env.pop(name, None)
        if student is not None:
            env["PDB_STUDENT_MODEL"] = student
        env.update(environ)
        command = ["bash", str(ROOT / launcher)]
        if launcher == "coding/eval_launcher.sh":
            command.append(str(task))
        result = subprocess.run(
            [*command, *args],
            env=env,
            cwd=root,
            capture_output=True,
            text=True,
        )
        calls = log.read_text().splitlines() if log.exists() else []
        return result, calls, root

    def test_defaults_to_tinker(self):
        """Without --server-backend, the launcher samples on Tinker."""
        result, calls, _ = self.run_launcher(
            "--split", "val", "--model-path", QWEN, "--limit", "1"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--backend tinker", calls[-1])
        self.assertIn(f"--model-path {QWEN}", calls[-1])

    def test_defaults_to_the_student_model(self):
        """Without --model-path, the launcher evaluates PDB_STUDENT_MODEL on Tinker.

        The old Qwen/Qwen3-4B default has no Tinker prices, so it failed in the
        sandbox before sampling anything.
        """
        for student, expected in ((QWEN, QWEN), (None, QWEN), ("Org/Other-1B", None)):
            with self.subTest(student=student):
                expected = expected or student
                result, calls, root = self.run_launcher(
                    "--split", "val", "--limit", "1", student=student
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(calls)
                for call in calls:  # --validate-only and the evaluation itself
                    self.assertIn(f"--model-path {expected}", call)
                    self.assertNotIn("Qwen3-4B", call)
                escaped = expected.replace("/", "__")
                self.assertIn(
                    f"--log-dir model_outputs/{escaped}_tinker/val", calls[-1]
                )
                # The baseline directory launch_agent copies into sandboxes.
                if expected == QWEN:
                    self.assertTrue(
                        (root / "task/model_outputs/Qwen__Qwen3-8B_tinker/val").is_dir()
                    )

    def test_host_defaults_to_tinker(self):
        """Outside the sandbox (no pin), the default backend is Tinker too."""
        result, calls, _ = self.run_launcher(
            "--split", "val", "--limit", "1", backend=""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--backend tinker", calls[-1])
        self.assertIn(
            f"--log-dir model_outputs/{QWEN.replace('/', '__')}_tinker/val", calls[-1]
        )

    def test_host_server_backend_variable_selects_a_local_server(self):
        """SERVER_BACKEND still picks a local server outside the sandbox."""
        result, calls, _ = self.run_launcher(
            "--split", "val", "--limit", "1", backend="", SERVER_BACKEND="sglang"
        )
        # No GPUs: it stops after checking that sglang is installed.
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(any("--backend tinker" in call for call in calls))
        self.assertTrue(any(call.endswith(" sglang") for call in calls), calls)

    def test_culture_keeps_its_local_default(self):
        """Culture's evaluator has no Tinker backend: its entry point stays local."""
        for name in ("evaluate_model.sh", "evaluate_normad.sh"):
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as out:
                    result, calls, _ = self.run_launcher(
                        "--split",
                        "val",
                        "--limit",
                        "1",
                        "--out-dir",
                        out,
                        backend="",
                        launcher=f"culture/{name}",
                    )
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertTrue(calls, result.stderr)
                self.assertIn("--validate-only", calls[0])
                self.assertFalse(any("--backend tinker" in call for call in calls))
                self.assertTrue(any(call.endswith(" vllm") for call in calls), calls)
        # The sandbox pin still wins, as before.
        with tempfile.TemporaryDirectory() as out:
            result, calls, _ = self.run_launcher(
                "--split",
                "val",
                "--limit",
                "1",
                "--out-dir",
                out,
                launcher="culture/evaluate_model.sh",
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--backend tinker", calls[-1])

    def test_local_backend_keeps_the_student_default(self):
        """Outside the sandbox (no pin), an explicit local backend still works."""
        result, calls, _ = self.run_launcher(
            "--split", "val", "--server-backend", "vllm", "--limit", "1", backend=""
        )
        # With no GPUs the launcher stops after validating the arguments, which
        # carry the student default; no server or evaluation starts.
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(any("--log-dir" in call for call in calls))
        self.assertTrue(calls, result.stderr)
        self.assertIn(f"--model-path {QWEN}", calls[0])
        self.assertIn("--validate-only", calls[0])

    def test_refuses_local_servers(self):
        """A local vLLM or sglang server is refused before anything starts."""
        for backend in ("vllm", "sglang"):
            with self.subTest(backend=backend):
                result, calls, _ = self.run_launcher(
                    "--split", "val", "--server-backend", backend, "--limit", "1"
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("Only --server-backend tinker", result.stderr)
                self.assertEqual(calls, [])


EVALUATOR_ARGS = r"""
import json, sys, types
sys.argv = ["evaluate_model.py", *json.loads(sys.argv[1])]
# Grading code (and its vendored grader checkout) is not needed to parse arguments.
checkers = types.ModuleType("checkers")
checkers.__getattr__ = lambda name: object  # e.g. checkers.CodeGenerationProblem
sys.modules["checkers"] = checkers
import evaluate_model
args = evaluate_model.parse_args()
print(json.dumps({"backend": args.backend, "model_path": args.model_path}))
"""


class EvaluatorDefaultTests(unittest.TestCase):
    """Calling evaluate_model.py directly follows the same pin and student default."""

    def parse(self, *argv, **environ):
        """Parse evaluator arguments in a subprocess under exactly these variables."""
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("PDB_EVAL_BACKEND", "PDB_STUDENT_MODEL")
        }
        env.update(environ)
        result = subprocess.run(
            [sys.executable, "-c", EVALUATOR_ARGS, json.dumps(argv)],
            env=env,
            cwd=ROOT / "coding",
            capture_output=True,
            text=True,
        )
        if result.returncode:
            return result
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_sandbox_defaults(self):
        """With the sandbox variables, the defaults are Tinker and the student."""
        parsed = self.parse(
            "--split", "val", PDB_EVAL_BACKEND="tinker", PDB_STUDENT_MODEL=QWEN
        )
        self.assertEqual(parsed, {"backend": "tinker", "model_path": QWEN})

    def test_host_defaults(self):
        """Without them, a local server and the base student, never Qwen3-4B."""
        parsed = self.parse("--split", "val")
        self.assertEqual(parsed, {"backend": "openai", "model_path": QWEN})

    def test_pinned_backend_refuses_a_local_server(self):
        """--backend openai is refused under the pin; merging shards is not sampling."""
        refused = self.parse(
            "--split", "val", "--backend", "openai", PDB_EVAL_BACKEND="tinker"
        )
        self.assertEqual(refused.returncode, 2)
        self.assertIn("only --backend tinker", refused.stderr)
        merged = self.parse(
            "--backend",
            "openai",
            "--merge-shards",
            "a.jsonl",
            PDB_EVAL_BACKEND="tinker",
        )
        self.assertEqual(merged["backend"], "openai")


class HostEvaluationTests(unittest.TestCase):
    """Host evaluation in Tinker mode samples on Tinker; legacy callers say so."""

    def test_secret_test_evaluation_pins_tinker(self):
        """final_evaluation names Tinker and pins it against a stray host export."""
        from coding.rl import agent_training

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"PDB_EVAL_BACKEND": "vllm", "SERVER_BACKEND": "sglang"}
        ), patch.object(agent_training, "run") as run:
            out = Path(directory) / "final/rl/.hidden/final_eval"
            agent_training.final_evaluation("tinker://run/sampler_weights/final", out)
        command, env = run.call_args.args[0], run.call_args.kwargs["env"]
        backend = command[command.index("--server-backend") + 1]
        self.assertEqual(backend, "tinker")
        self.assertEqual(env["PDB_EVAL_BACKEND"], "tinker")
        self.assertEqual(env["TINKER_EVAL_BUDGET"], "eval")
        self.assertEqual(
            command[command.index("--model-path") + 1],
            "tinker://run/sampler_weights/final",
        )

    def test_secret_test_evaluation_uses_this_interpreter(self):
        """eval_launcher.sh's bare `python` is this interpreter, not the shell's."""
        from coding.rl import agent_training

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"PATH": "/usr/bin"}
        ), patch.object(agent_training, "run") as run:
            out = Path(directory) / "final/rl/.hidden/final_eval"
            agent_training.final_evaluation("tinker://run/sampler_weights/final", out)
        path = run.call_args.kwargs["env"]["PATH"].split(os.pathsep)
        self.assertEqual(path, [str(Path(sys.executable).parent), "/usr/bin"])

    def test_local_checkpoint_callers_name_their_backend(self):
        """Every caller that evaluates a local checkpoint names a local backend."""
        callers = {
            "coding/run_model_training_and_evaluation.sh": '"${SERVER_BACKEND:-vllm}"',
            "coding/rl/eval_rl_checkpoint.sh": '"${SERVER_BACKEND:-vllm}"',
            "coding/rl/run_rl_training_and_evaluation.sh": "sglang",
        }
        for name, backend in callers.items():
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                call = source[source.rindex("./evaluate_model.sh") :]
                self.assertIn(f"--server-backend {backend}", " ".join(call.split()))
        # run_evaluate_only.sh exports its sglang default for evaluate_model.sh.
        for task in ("coding", "culture"):
            source = (ROOT / task / "run_evaluate_only.sh").read_text()
            self.assertIn('export SERVER_BACKEND="${SERVER_BACKEND:-sglang}"', source)


PARITY = r"""
import asyncio, json, sys, types
from types import SimpleNamespace

calls = []
SPECIAL = {151645: "<|im_end|>", 151643: "<|endoftext|>"}

class Tokenizer:
    eos_token_id = 151645
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        assert tokenize is False and add_generation_prompt is True
        text = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        return text + "<|im_start|>assistant\n" + json.dumps(kw, sort_keys=True)
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [ord(c) for c in text]
    def decode(self, tokens, skip_special_tokens=False):
        return "".join(
            "" if t in SPECIAL and skip_special_tokens else SPECIAL.get(t) or chr(t)
            for t in tokens
        )

class Future:
    def __init__(self, value):
        self.value = value
    def result(self):
        return self.value

class Sampler:
    def __init__(self, path, base):
        self.path, self.base = path, base
    def get_base_model(self):
        return "Qwen/Qwen3-8B"
    def sample(self, prompt, num_samples, sampling_params):
        calls.append({"prompt": prompt.tokens, "n": num_samples,
                      "params": sampling_params.kwargs, "path": self.path})
        text = "<think>t</think>answer"
        seq = SimpleNamespace(tokens=[ord(c) for c in text] + [151645],
                              stop_reason="stop")
        return Future(SimpleNamespace(sequences=[seq] * num_samples,
                                      prompt_cache_hit_tokens=0))
    async def sample_async(self, **kwargs):
        return self.sample(**kwargs).result()

class Service:
    def __init__(self, user_metadata=None):
        self.holder = SimpleNamespace(_session_id="s")
    def create_sampling_client(self, model_path=None, base_model=None):
        return Sampler(model_path, base_model)
    def get_server_capabilities(self):
        return SimpleNamespace(supported_models=[SimpleNamespace(
            model_name="Qwen/Qwen3-8B", max_context_length=int(sys.argv[2]))])

class Params:
    def __init__(self, **kw):
        self.kwargs = kw

class ModelInput:
    def __init__(self, tokens):
        self.tokens = tokens
    @classmethod
    def from_ints(cls, tokens):
        return cls(list(tokens))

tinker = types.ModuleType("tinker")
tinker.ServiceClient = Service
tinker.types = SimpleNamespace(SamplingParams=Params, ModelInput=ModelInput)
transformers = types.ModuleType("transformers")
transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda name: Tokenizer())
transformers.GenerationConfig = SimpleNamespace(
    from_pretrained=lambda name: SimpleNamespace(eos_token_id=[151645, 151643]))
sys.modules["tinker"], sys.modules["transformers"] = tinker, transformers

import pyforger
assert pyforger.__file__.startswith(sys.argv[3]), pyforger.__file__
from pyforger.acquire import TinkerModel
from tinker_sampling import TinkerChat

messages = [{"role": "system", "content": "You are an expert."},
            {"role": "user", "content": "Solve it."}]
path = "tinker://run/sampler_weights/final"
thinking = json.loads(sys.argv[1])
args = SimpleNamespace(model_path=path, max_tokens=28672, temperature=0.6,
                       top_p=0.95, top_k=20, min_p=0, enable_thinking=thinking,
                       log_dir=sys.argv[4])
completion, _, _ = asyncio.run(TinkerChat(args)(messages))
model = TinkerModel(path, max_tokens=28672, enable_thinking=thinking)
(result,) = model.generate_batch([messages])
calls[1]["params"].pop("seed")
print(json.dumps({"calls": calls, "texts": [completion, result[0]]}))
"""


class EvaluatorParityTests(unittest.TestCase):
    """TinkerModel sends what the evaluator's TinkerChat sends."""

    def test_same_request_as_the_evaluator(self):
        """Prompt tokens, stop ids, clamped max_tokens and sampling settings match."""
        for thinking, context in ((None, 32768), (False, 200)):
            with self.subTest(
                thinking=thinking
            ), tempfile.TemporaryDirectory() as directory:
                env = {
                    **os.environ,
                    "PYTHONPATH": f"{REPO / 'src'}{os.pathsep}{ROOT / 'coding'}",
                }
                for name in (
                    "TINKER_EVAL_BUDGET",
                    "FORGER_TINKER_USAGE_FILE",
                    "FORGER_TINKER_USD_LIMIT",
                ):
                    env.pop(name, None)
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        PARITY,
                        json.dumps(thinking),
                        str(context),
                        str(REPO / "src"),
                        directory,
                    ],
                    env=env,
                    cwd=directory,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                output = json.loads(result.stdout.strip().splitlines()[-1])
                chat, forger = output["calls"]
                self.assertEqual(chat, forger)
                self.assertEqual(chat["params"]["stop"], [151645, 151643])
                if context == 200:
                    self.assertLess(chat["params"]["max_tokens"], 200)
                texts = output["texts"]
                self.assertEqual(texts[0], texts[1])
                self.assertEqual(texts[0], "<think>t</think>answer")


if __name__ == "__main__":
    unittest.main()
