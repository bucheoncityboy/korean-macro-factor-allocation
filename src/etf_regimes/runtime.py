"""macOS wheel compatibility, using only the OpenMP runtime in the locked sklearn wheel."""
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


def prepare_openmp():
    if sys.platform != "darwin":
        return
    sklearn_spec = importlib.util.find_spec("sklearn")
    xgb_spec = importlib.util.find_spec("xgboost")
    runtime = Path(sklearn_spec.origin).parent / ".dylibs/libomp.dylib"
    library = Path(xgb_spec.origin).parent / "lib/libxgboost.dylib"
    if not runtime.exists():
        raise RuntimeError("Locked scikit-learn wheel has no bundled macOS OpenMP runtime")
    dependencies = subprocess.check_output(["otool", "-L", str(library)], text=True)
    if "@rpath/libomp.dylib" in dependencies:
        # Break uv's cache hardlink before editing a local dependency. Relative linkage is portable.
        temporary = library.with_suffix(".local.dylib")
        shutil.copy2(library, temporary)
        subprocess.run(["install_name_tool", "-change", "@rpath/libomp.dylib",
                        "@loader_path/../../sklearn/.dylibs/libomp.dylib", str(temporary)], check=True)
        temporary.replace(library)
