from __future__ import annotations

from pathlib import Path

from skyfield.api import load


def main() -> None:
    target = Path("assets") / "ephemeris" / "de421.bsp"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"Ephemeris already present: {target}")
        return
    eph = load("de421.bsp")
    target.write_bytes(Path(eph.path).read_bytes())
    print(f"Downloaded ephemeris to {target}")


if __name__ == "__main__":
    main()