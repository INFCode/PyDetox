from itertools import accumulate, tee, chain
from typing import List, Optional, Tuple, TypeVar
from pathlib import Path
import functools
from copy import deepcopy

DEBUG = True

T = TypeVar("T")


def flatten(lst: List[List[T]]) -> Tuple[List[T], List[Tuple[int, int]]]:
    output = sum(lst, [])
    accumulated_len = accumulate(map(len, lst))
    a, b = tee(chain([0], accumulated_len))
    next(b, None)
    return output, list(zip(a, b))


def unflatten(lst: List[T], partition: List[Tuple[int, int]]) -> List[List[T]]:
    return [lst[a:b] for a, b in partition]


def find_project_root(current_path: Optional[Path | str] = None) -> Path:
    """
    Finds the project's root directory by looking for a directory that contains
    a .git folder, a requirements.txt, or a LICENSE file.
    """
    if current_path is None:
        # If no path is provided, use the directory of the current file (__file__).
        current_path = Path(__file__).resolve().parent
    else:
        # Ensure current_path is a Path object
        current_path = Path(current_path).resolve()

    # Traverse up until we find the root directory
    for path in [current_path, *current_path.parents]:
        if any(
            (path / marker).exists()
            for marker in [".git", "requirements.txt", "LICENSE"]
        ):
            return path

    # If we haven't found a project root, raise an error
    raise FileNotFoundError("No project root found")


def relative_to_project_root(
    relative_path: Path | str, current_path: Optional[Path | str] = None
) -> Path:
    """
    Returns the absolute path to a file given its relative path from the project root.

    Parameters:
    - relative_path (str): The relative path to the file from the project root.
    - current_path (Optional[Path]): The starting path to use for finding the project root.
                                     If None, defaults to the directory of the current file.

    Returns:
    - Path: The absolute path to the target file.
    """
    # Find the project root
    project_root = find_project_root(current_path)

    # Create the full path by appending the relative path to the project root
    full_path = project_root / relative_path

    # Return the full path, making sure to resolve it to an absolute path
    return full_path.resolve()


def lru_cache(maxsize=128, typed=False, copy=False):
    """
    a simple wrapper of functools.lru_cache, to support deep copy
    """
    if not copy:
        return functools.lru_cache(maxsize, typed)

    def decorator(f):
        cached_func = functools.lru_cache(maxsize, typed)(f)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return deepcopy(cached_func(*args, **kwargs))

        return wrapper

    return decorator
