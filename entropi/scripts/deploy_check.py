"""Pre-deploy checklist. Run this before pushing to Railway."""

import sys
import os
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

checks_passed = 0
checks_failed = 0


def check(name, condition):
    global checks_passed, checks_failed
    if condition:
        print(f"  OK  {name}")
        checks_passed += 1
    else:
        print(f"  FAIL  {name}")
        checks_failed += 1


# files exist
check("Dockerfile exists", (root / "Dockerfile").exists())
check("requirements.txt exists", (root / "requirements.txt").exists())
check("railway.json exists", (root / "railway.json").exists())

# .env.example has required vars
env_example = root / ".env.example"
if env_example.exists():
    content = env_example.read_text()
    check(".env.example has DATABASE_URL", "DATABASE_URL" in content)
    check(".env.example has API_ENV", "API_ENV" in content)
    check(".env.example has LOG_LEVEL", "LOG_LEVEL" in content)
else:
    check(".env.example exists", False)

# imports work
try:
    import entropi
    check("entropi package imports", True)
except Exception as e:
    check(f"entropi package imports ({e})", False)

try:
    from entropi.api.main import app
    check("entropi.api.main.app imports", True)
except Exception as e:
    check(f"entropi.api.main.app imports ({e})", False)

# summary
print()
if checks_failed == 0:
    print(f"  {checks_passed}/{checks_passed} checks passed. Ready to deploy!")
else:
    print(f"  {checks_failed} check(s) failed. Fix the errors above first.")
    sys.exit(1)
