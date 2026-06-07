"""Allow ``python -m research_navigator.ingest`` to invoke the ingest CLI."""

from __future__ import annotations

import sys

from research_navigator.cli.ingest import main

if __name__ == "__main__":
    sys.exit(main())
