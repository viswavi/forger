"""Tests for Forger core infrastructure."""

import pytest
from fixtures import (
    make_concat,
    make_double_repetition_data,
    make_empty_filter,
    make_source,
)

from pyforger import DataSpec, NodeType, run


class TestGraphConstruction:
    """Operations build graph nodes without executing."""

    def test_returns_dataspec(self):
        """Test that operations return DataSpec nodes."""
        assert isinstance(make_source([1, 2, 3]), DataSpec)

    def test_no_execution_on_build(self):
        """Test that execute function is not called on node creation."""
        execution_count = 0

        def execute():
            nonlocal execution_count
            execution_count += 1
            return [1, 2, 3]

        DataSpec(NodeType.ACQUIRE, [], execute, "test_source")
        assert execution_count == 0

    def test_single_dependency_tracked(self):
        """Test that nodes can track one upstream dependency.

        Create a node that depends on another, and check that this dependency
        is recorded.
        """
        source = make_source([1, 2, 3])
        doubled = make_double_repetition_data(source)
        assert doubled.get_dependencies() == [source]

    def test_multiple_dependencies_tracked(self):
        """Test that nodes can track multiple upstream dependencies.

        Create a node that depends on multiple others, and check that these dependencies
        are recorded.
        """
        a, b = make_source([1]), make_source([2])
        combined = make_concat(a, b)
        assert set(combined.get_dependencies()) == {a, b}


class TestExecution:
    """run() executes the graph correctly."""

    def test_simple(self):
        """Executing a simple, one-node graph gives the expected result."""
        assert run(make_source([1, 2, 3])) == [1, 2, 3]

    def test_chain(self):
        """Executing a chain of two nodes gives the expected result."""
        result = run(make_double_repetition_data(make_source([1, 2])))
        assert result == [1, 2, 1, 2]

    def test_caching_prevents_reexecution(self):
        """Running nodes that were already run does not re-execute them."""
        execution_count = 0

        def execute():
            nonlocal execution_count
            execution_count += 1
            return [1, 2]

        source = DataSpec(NodeType.ACQUIRE, [], execute, "test_source")
        doubled = make_double_repetition_data(source)
        run(doubled)
        run(doubled)
        assert execution_count == 1

    def test_diamond_executes_shared_node_once(self):
        """Test that a shared dependency in a diamond graph is only executed once."""
        execution_count = 0

        def execute():
            nonlocal execution_count
            execution_count += 1
            return [1, 2]

        source = DataSpec(NodeType.ACQUIRE, [], execute, "test_source")
        merged = make_concat(
            make_double_repetition_data(source), make_double_repetition_data(source)
        )
        run(merged)
        assert execution_count == 1

    def test_empty_filter_returns_empty_list(self):
        """Test that a filter yielding no matches returns an empty list."""
        filtered = make_empty_filter(make_source([1, 2, 3]))
        assert run(filtered) == []

    def test_pipeline_defined_as_function(self):
        """Test that a pipeline defined as a function executes normally."""

        def pipeline():
            return make_source([1, 2, 3])

        assert run(pipeline) == [1, 2, 3]


class TestErrorHandling:
    """Errors during execution are handled properly."""

    def test_node_throws_not_implemented_error(self):
        """Running a node that raises NotImplementedError, leads to a thrown error."""

        def _throw_error():
            raise NotImplementedError("Pipeline failed")

        node_throwing_err = DataSpec(
            NodeType.ACQUIRE, [], _throw_error, "test_failing_node"
        )
        with pytest.raises(NotImplementedError):
            run(node_throwing_err)

    def test_result_before_run_raises(self):
        """Test that a pipeline's result cannot be accessed before it has run."""
        with pytest.raises(RuntimeError, match="not been executed"):
            _ = make_source([1]).result
