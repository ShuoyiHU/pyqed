"""Click-run 2D bose_hubbard comparison."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from compare import main as compare

INCLUDE_TWO_SITE = True  # Set False for IDE runs, or pass --skip-two-site.


def main(argv=None):
    return compare(argv, model="bose_hubbard", include_two_site=INCLUDE_TWO_SITE)


if __name__ == "__main__":
    main()
