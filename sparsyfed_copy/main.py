"""Module entry point to launch SparsyFed experiments."""

from project.main import main as project_main


def main() -> None:
    """Delegate execution to :mod:`project.main`."""
    project_main()


if __name__ == "__main__":
    main()
