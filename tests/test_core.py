"""Tests for Forger core infrastructure."""

from fixtures import CallCounter, make_concat, make_double_repetition_data, make_source

from pyforger import DataSpec


class TestGraphConstruction:
    """Operations build graph nodes without executing."""

    def test_returns_dataspec(self):
        """Test that operations return DataSpec nodes."""
        assert isinstance(make_source([1, 2, 3]), DataSpec)

    def test_no_execution_on_build(self):
        """Test that execute function is not called on node creation."""
        counter = CallCounter()
        make_source([1, 2, 3], counter=counter)
        assert counter.count == 0

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
