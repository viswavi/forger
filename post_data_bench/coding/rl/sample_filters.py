"""Slime rollout filters; use rl.sample_filters.drop_truncated.

Set remove_sample for a zero loss mask. Slime still includes masked samples
in advantage normalization. Helpers do not require importing Slime."""
from __future__ import annotations

import sys

# The value behind slime's Sample.Status.TRUNCATED, i.e. a completion that hit
# --rollout-max-response-len. Comparing on the value rather than importing the
# enum keeps this module importable without slime.
_TRUNCATED = "truncated"


def _iter_samples(obj):
    """Walk a rollout payload, which arrives as groups nested one or two deep."""
    if isinstance(obj, list):
        for item in obj:
            yield from _iter_samples(item)
    else:
        yield obj


def drop_truncated(args, samples, **kwargs) -> None:
    """Mask capped completions without changing rewards or advantage normalization.

    Truncation can bias learning against answers reached late, such as abstention.
    Masking removes that gradient but also removes pressure against long responses;
    monitor response length. Malformed payloads are reported and skipped because
    filter errors must not interrupt training."""
    total = dropped = 0
    try:
        for sample in _iter_samples(samples):
            status = getattr(sample, "status", None)
            total += 1
            if getattr(status, "value", status) == _TRUNCATED:
                sample.remove_sample = True
                dropped += 1
        note = ""
    except Exception as e:
        note = (
            f" -- STOPPED EARLY after {total} ({type(e).__name__}: {e}); "
            "the rest of this rollout trains unfiltered"
        )
    print(
        f"[sample_filters] drop_truncated: masked {dropped}/{total} truncated samples{note}",
        file=sys.stderr,
        flush=True,
    )
