from __future__ import annotations

from . import _ensure_src_on_path

_ensure_src_on_path()

from cli.commands import main


if __name__ == "__main__":
    main()
