"""Entry point of a data-parallel embedding worker: `python -m ... FD`.

Kept apart from `_embed_dp` (which the package imports) so `python -m` does not
run a module that is already in `sys.modules`.
"""

import sys
from multiprocessing.connection import Connection

from pyforger.inspect._embed_dp import serve

# Importing this module, e.g. by pkgutil.walk_packages, must not start a worker.
if __name__ == "__main__":
    serve(Connection(int(sys.argv[1])))
