"""Download the fingerprinted public Hotel Booking Demand source."""

from pathlib import Path
import hashlib
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
    "master/data/2020/2020-02-11/hotels.csv"
)
EXPECTED_SHA256 = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"


def main() -> None:
    payload = urlopen(URL, timeout=60).read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(
            f"Source changed: expected {EXPECTED_SHA256}, got {digest}. "
            "Review before updating the reference."
        )
    target = ROOT / "data" / "raw" / "hotel_bookings.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    print(f"Saved {target}; SHA-256 {digest}")


if __name__ == "__main__":
    main()
