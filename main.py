"""Convenience entry point for running SparsyFed experiments.

This script imports :func:`project.main.main` so that the project can be
started with ``python main.py`` from the repository root.
"""

from project.main import main as project_main


if __name__ == "__main__":
    project_main()
