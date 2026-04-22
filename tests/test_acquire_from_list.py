"""Tests for acquire.from_list."""
from __future__ import annotations

from pyforger import run
from pyforger.acquire import from_list
from pyforger.core import NodeType


class TestFromList:
    """from_list wraps an in-memory list as an ACQUIRE DataSpec."""

    def test_returns_items_unchanged(self):
        """The DataSpec's output equals the input list."""
        data = [{"x": 1}, {"x": 2}, {"x": 3}]
        spec = from_list(data)
        assert run(spec) == data

    def test_empty_list(self):
        """Empty input returns empty output."""
        spec = from_list([])
        assert run(spec) == []

    def test_node_type_is_acquire(self):
        """from_list builds an ACQUIRE DataSpec."""
        spec = from_list([])
        assert spec.node_type == NodeType.ACQUIRE

    def test_default_name(self):
        """Default node name is 'from_list'."""
        spec = from_list([])
        assert spec.name == "from_list"

    def test_custom_name(self):
        """Custom name overrides the default."""
        spec = from_list([], name="seed_pool")
        assert spec.name == "seed_pool"

    def test_independent_specs_keep_separate_data(self):
        """Two from_list calls in the same scope return distinct lists."""
        a = from_list([{"v": "a"}])
        b = from_list([{"v": "b"}])
        assert run(a) == [{"v": "a"}]
        assert run(b) == [{"v": "b"}]

    def test_specs_built_in_loop_capture_correctly(self):
        """Each from_list call in a loop captures its own items, not the last."""
        specs = [from_list([{"i": i}]) for i in range(3)]
        results = [run(s) for s in specs]
        assert results == [[{"i": 0}], [{"i": 1}], [{"i": 2}]]

    def test_no_dependencies(self):
        """from_list nodes have no upstream inputs."""
        spec = from_list([{"x": 1}])
        assert spec.get_dependencies() == []

    def test_list_captured_by_reference(self):
        """Mutating the source list after construction shows up at run time."""
        data: list[dict] = []
        spec = from_list(data)
        data.append({"added": True})
        assert run(spec) == [{"added": True}]
