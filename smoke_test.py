# Run from your project root:  python smoke_test.py
from pathlib import Path
from src.data_load import _as_path_or_url

s = _as_path_or_url("  https://example.com/file.csv  ")
assert isinstance(s, str) and s == "https://example.com/file.csv"

try:
    _as_path_or_url("https://")
    raise AssertionError("Expected ValueError for malformed URL")
except ValueError:
    pass

try:
    _as_path_or_url("   ")
    raise AssertionError("Expected ValueError for empty input")
except ValueError:
    pass

# Windows drive path (forward slashes or backslashes)
p1 = _as_path_or_url("C:/tmp/file.csv")
p2 = _as_path_or_url(r"C:\tmp\file.csv")
assert isinstance(p1, Path) and isinstance(p2, Path)

# UNC path
p3 = _as_path_or_url(r"\\server\share\file.csv")
assert isinstance(p3, Path)

# file:// URL
p4 = _as_path_or_url("file:///C:/tmp/file.csv")
assert isinstance(p4, Path)

print("data_load URL/path handling OK")
