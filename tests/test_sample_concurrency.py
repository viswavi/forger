import threading

import pytest

from pyforger.alter import sample


def test_sample_overlaps_calls_without_reordering():
    barrier = threading.Barrier(3)
    barrier.cache_key = lambda: "concurrency-test-barrier"

    def generate(item, identity):
        barrier.wait(timeout=10)
        return (item["index"], identity)

    rows = [{"index": i} for i in range(3)]
    result = sample.no_cache(rows, generate, sample_ids=["one"], concurrency=3)
    assert [row["samples"][0]["value"] for row in result] == [
        (i, "one") for i in range(3)
    ]
    assert rows == [{"index": i} for i in range(3)]


@pytest.mark.parametrize("concurrency", [0, -1, True])
def test_sample_rejects_invalid_concurrency(concurrency):
    with pytest.raises(ValueError):
        sample.no_cache(
            [], lambda *_: None, sample_ids=["one"], concurrency=concurrency
        )
