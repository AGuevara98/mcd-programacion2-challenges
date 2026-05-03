from __future__ import annotations

import datetime
import subprocess


def main() -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(
            ["git", "commit", "-m", f"Challenge results — {timestamp}"],
            check=True,
        )
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("Results uploaded successfully.")
    except subprocess.CalledProcessError as exc:
        print(f"Git operation failed: {exc}")
        raise


if __name__ == "__main__":
    main()
