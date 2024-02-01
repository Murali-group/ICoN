'''MIT License

Copyright (c) 2020 Duncan Forster

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''


import typer
from pathlib import Path
import torch


def extend_path(path: Path, extension: str) -> Path:
    """Extends a path by adding an extension to the stem.

    Args:
        path (Path): Full path.
        extension (str): Extension to add. This will replace the current extension.

    Returns:
        Path: New path with extension.
    """
    return path.parent / (path.stem + extension)


def cyan(string: str, **kwargs) -> str:
    return typer.style(string, fg=typer.colors.CYAN, bold=True, **kwargs)


def magenta(string: str, **kwargs) -> str:
    return typer.style(string, fg=typer.colors.MAGENTA, bold=True, **kwargs)


class Device:
    """Returns the currently used device by calling `Device()`.

    Returns:
        str: Either "cuda" or "cpu".
    """

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    def __new__(cls) -> str:
        return cls._device


