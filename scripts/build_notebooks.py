"""Convert the percent-format notebook companions using Jupytext; optionally execute."""
import argparse
import os
import sys
from pathlib import Path
import jupytext
import nbformat
from nbclient import NotebookClient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    for source in sorted((root / "notebooks").glob("*.py")):
        nb = jupytext.read(source)
        nb.metadata["kernelspec"] = dict(display_name="Python 3", language="python", name="python3")
        nb.metadata["jupytext"] = dict(formats="ipynb,py:percent")
        nbformat.validate(nb)
        if a.execute:
            NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
        target = source.with_suffix(".ipynb")
        jupytext.write(nb, target)
        print(target.name, "executed" if a.execute else "converted", flush=True)


if __name__ == "__main__":
    main()
