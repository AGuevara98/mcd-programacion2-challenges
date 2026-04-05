import subprocess
import sys


def run_command(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    print(result.stdout)


def main():
    commit_message = "Resultados del challenge"
    if len(sys.argv) > 1:
        commit_message = sys.argv[1]

    run_command(["git", "add", "."])
    run_command(["git", "commit", "-m", commit_message])
    run_command(["git", "push", "origin", "main"])


if __name__ == "__main__":
    main()