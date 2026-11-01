"""Object generation validates real oracle artifacts across both retry budgets."""

import json
import shlex

import pytest

from pyforger import agents
from pyforger.agent_backend import AgentSpec
from pyforger.local_backend import LocalBackend
from pyforger.structured import ValidationExhausted


class RecordingBackend(LocalBackend):
    """Observe workspace requests without replacing backend execution."""

    def __init__(self):
        """Collect fresh-generation task requests."""
        self.tasks = []

    def run_agent(self, task, agent):
        """Record unvalidated fresh generations."""
        self.tasks.append(task)
        return super().run_agent(task, agent)

    def run_agent_validated(self, task, agent, **kwargs):
        """Record validated sessions while executing real oracle scripts."""
        self.tasks.append(task)
        return super().run_agent_validated(task, agent, **kwargs)


def parse(files):
    """Decode JSON and enforce a small caller-owned domain constraint."""
    value = json.loads(files["answer.json"])
    if not isinstance(value, dict) or value.get("accepted") is not True:
        raise ValueError("domain rejection: accepted must be true")
    return value


@pytest.mark.parametrize("inner", [False, True])
def test_outer_retry_retains_rejections(inner):
    """Fresh generations receive raw rejected files and exact feedback."""
    backend = RecordingBackend()
    result = agents.generate_object(
        "Produce answer.json",
        {"input": "original"},
        "answer.json",
        parse=parse,
        agent=AgentSpec(
            kind="oracle",
            solution=(
                "if test -f previous-1/answer.json; then "
                "printf '{\"accepted\":true}' > answer.json; "
                "else printf 'broken JSON' > answer.json; fi"
            ),
        ),
        backend=backend,
        inner_loop_validation=inner,
        max_repairs=0,
    )
    assert result == {"accepted": True}
    assert len(backend.tasks) == 2
    assert backend.tasks[1].files["previous-1/answer.json"] == b"broken JSON"
    assert backend.tasks[1].files["input"] == "original"
    assert "Expecting value: line 1 column 1 (char 0)" in backend.tasks[1].instruction


def test_inner_repair_reuses_workspace():
    """Same-session repairs finish before another fresh generation is spent."""
    backend = RecordingBackend()
    result = agents.generate_object(
        "Produce answer.json",
        {},
        "answer.json",
        parse=parse,
        agent=AgentSpec(
            kind="oracle",
            solution=(
                "if test -f attempted; then "
                "printf '{\"accepted\":true}' > answer.json; "
                "else touch attempted; printf '{}' > answer.json; fi"
            ),
        ),
        backend=backend,
        max_attempts=1,
        max_repairs=1,
    )
    assert result == {"accepted": True}
    assert len(backend.tasks) == 1


@pytest.mark.parametrize("inner", [False, True])
def test_exhaustion_retains_latest_raw_output(inner):
    """Exhaustion raises structured failure rather than a partial parsed value."""
    backend = RecordingBackend()
    with pytest.raises(ValidationExhausted) as caught:
        agents.generate_object(
            "Produce answer.json",
            {},
            "answer.json",
            parse=parse,
            agent=AgentSpec(kind="oracle", solution="echo '{}' > answer.json"),
            backend=backend,
            inner_loop_validation=inner,
            max_attempts=2,
            max_repairs=1,
        )
    assert len(backend.tasks) == 2
    assert len(caught.value.attempts) == 2
    assert caught.value.output == {"answer.json": b"{}\n"}
    assert caught.value.measurement.reason == "domain rejection: accepted must be true"


@pytest.mark.parametrize("inner", [False, True])
def test_missing_output_is_rejected(inner):
    """Absent outputs use validation budgets, not infrastructure failure handling."""
    with pytest.raises(ValidationExhausted, match="Missing declared outputs"):
        agents.generate_object(
            "Produce answer.json",
            {},
            "answer.json",
            parse=parse,
            agent=AgentSpec(kind="nop"),
            inner_loop_validation=inner,
            max_attempts=1,
            max_repairs=0,
        )


def test_unexpected_parser_failure_propagates():
    """Runtime errors in domain code do not trigger repair attempts."""
    backend = RecordingBackend()

    def broken(files):
        raise RuntimeError("domain implementation bug")

    with pytest.raises(RuntimeError, match="domain implementation bug"):
        agents.generate_object(
            "Produce answer.json",
            {},
            "answer.json",
            parse=broken,
            agent=AgentSpec(kind="oracle", solution="echo '{}' > answer.json"),
            backend=backend,
        )
    assert len(backend.tasks) == 1


@pytest.mark.parametrize(
    "kwargs", [{"max_attempts": 0}, {"max_attempts": True}, {"max_repairs": -1}]
)
def test_invalid_budgets_do_not_launch(kwargs):
    """Budget configuration is checked before any backend work."""
    backend = RecordingBackend()
    with pytest.raises(ValueError):
        agents.generate_object(
            "Produce answer.json",
            {},
            "answer.json",
            parse=parse,
            agent=AgentSpec(kind="nop"),
            backend=backend,
            **kwargs,
        )
    assert backend.tasks == []


@pytest.mark.parametrize("inner", [False, True])
def test_none_is_a_valid_parsed_result(inner):
    """Parser results need not be truthy or non-null to count as accepted."""
    assert (
        agents.generate_object(
            "Produce output",
            {},
            "output",
            parse=lambda files: None,
            agent=AgentSpec(kind="oracle", solution="touch output"),
            inner_loop_validation=inner,
        )
        is None
    )


def test_previous_snapshot_never_overwrites_input():
    """Colliding retained paths fail explicitly instead of replacing user data."""
    with pytest.raises(ValueError, match="conflicts with input"):
        agents.generate_object(
            "Produce answer.json",
            {"previous-1/answer.json": "original"},
            "answer.json",
            parse=parse,
            agent=AgentSpec(kind="oracle", solution="echo '{}' > answer.json"),
            inner_loop_validation=False,
        )


@pytest.mark.parametrize("inner", [False, True])
def test_exact_combined_budgets_and_all_outer_snapshots(tmp_path, inner):
    """Inner repairs multiply turns, not fresh attempts; rejected inputs accumulate."""
    backend = RecordingBackend()
    log = shlex.quote(str(tmp_path / "turns"))
    with pytest.raises(ValidationExhausted):
        agents.generate_object(
            "Produce answer.json",
            {},
            "answer.json",
            parse=parse,
            agent=AgentSpec(
                kind="oracle", solution=f"echo turn >> {log}; echo '{{}}' > answer.json"
            ),
            backend=backend,
            inner_loop_validation=inner,
            max_attempts=3,
            max_repairs=2,
        )
    assert len((tmp_path / "turns").read_text().splitlines()) == (9 if inner else 3)
    assert len(backend.tasks) == 3
    assert backend.tasks[2].files == {
        "previous-1/answer.json": b"{}\n",
        "previous-2/answer.json": b"{}\n",
    }


@pytest.mark.parametrize("inner", [False, True])
def test_execution_errors_are_not_retried(inner):
    """Oracle command errors never consume output-validation retry budgets."""
    backend = RecordingBackend()
    with pytest.raises(RuntimeError):
        agents.generate_object(
            "Produce answer.json",
            {},
            "answer.json",
            parse=parse,
            agent=AgentSpec(kind="oracle", solution="exit 2"),
            backend=backend,
            inner_loop_validation=inner,
        )
    assert len(backend.tasks) == 1


@pytest.mark.parametrize("inner", [False, True])
def test_additional_outputs_are_collected(inner):
    """Declared extra outputs are collected; the named output stays required."""
    backend = RecordingBackend()

    def parse_both(files):
        assert json.loads(files["staged.json"]) == {"value": 2}
        return parse(files)

    result = agents.generate_object(
        "Produce answer.json",
        {"staged.json": '{"value": 1}', "seed.txt": "seed"},
        "answer.json",
        parse=parse_both,
        agent=AgentSpec(
            kind="oracle",
            solution=(
                "printf '{\"value\": 2}' > staged.json; "
                "printf '{\"accepted\":true}' > answer.json"
            ),
        ),
        backend=backend,
        inner_loop_validation=inner,
        max_repairs=0,
        outputs=("answer.json", "staged.json"),
    )
    assert result == {"accepted": True}
    assert backend.tasks[0].outputs == ("answer.json", "staged.json")

    # The named output is still the only required one: extras may be absent.
    with pytest.raises(ValidationExhausted, match="Missing declared outputs"):
        agents.generate_object(
            "Produce answer.json",
            {},
            "answer.json",
            parse=parse,
            agent=AgentSpec(kind="nop"),
            backend=RecordingBackend(),
            inner_loop_validation=inner,
            max_attempts=1,
            max_repairs=0,
            outputs=("answer.json", "staged.json"),
        )

