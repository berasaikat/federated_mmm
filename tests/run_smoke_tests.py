import subprocess
import sys
from pathlib import Path

tests_dir = Path(__file__).parent

# Define phases and their smoke tests in order
smoke_tests = [
    ("Phase 3 — MMM",              "smoke_test_phase3.py"),
    ("Phase 4.1 — Prompt Builder", "smoke_test_phase4_1_prompt_builder.py"),
    ("Phase 4.2 — Validator",      "smoke_test_phase4_2_validator.py"),
    ("Phase 4.3 — Surprise",       "smoke_test_phase4_3_surprise.py"),
    ("Phase 4.4 — API Call",       "smoke_test_phase4_4_api_call.py"),
    ("Phase 4.5 — Feedback Loop",  "smoke_test_phase4_5_feedback_loop.py"),
]

results = []

for label, filename in smoke_tests:
    filepath = tests_dir / filename
    if not filepath.exists():
        results.append((label, "SKIPPED", "file not found"))
        continue

    print(f"\n{'='*50}")
    print(f"Running: {label}")
    print(f"{'='*50}")

    result = subprocess.run(
        [sys.executable, str(filepath)],
        capture_output=False   # streams output live
    )

    status = "PASSED" if result.returncode == 0 else "FAILED"
    results.append((label, status, ""))

# Print summary
print(f"\n{'='*50}")
print("SMOKE TEST SUMMARY")
print(f"{'='*50}")
for label, status, note in results:
    icon = "✓" if status == "PASSED" else "✗" if status == "FAILED" else "–"
    note_str = f"  ({note})" if note else ""
    print(f"  {icon} {status:<8} {label}{note_str}")

# Exit with error if any failed
failed = [r for r in results if r[1] == "FAILED"]
if failed:
    print(f"\n{len(failed)} test(s) failed.")
    sys.exit(1)
else:
    print(f"\nAll tests passed.")