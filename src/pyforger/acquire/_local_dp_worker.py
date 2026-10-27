"""Entry point of a `LocalModel` data-parallel worker: `python -m ... FD`.

Kept apart from `_local_dp` (which the package imports) so `python -m` does not
run a module that is already in `sys.modules`.
"""

import sys
from multiprocessing.connection import Connection

from pyforger.acquire._local_dp import serve

serve(Connection(int(sys.argv[1])))
