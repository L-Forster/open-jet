from __future__ import annotations

from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_py import build_py as _build_py


PACKAGE_NAME = "src"
PACKAGE_DIR = Path(PACKAGE_NAME)
EXCLUDE_FROM_CYTHON = {"__init__.py"}


def iter_extension_modules() -> list[Extension]:
    modules: list[Extension] = []
    for path in PACKAGE_DIR.glob("*.py"):
        if path.name in EXCLUDE_FROM_CYTHON:
            continue
        modules.append(Extension(f"{PACKAGE_NAME}.{path.stem}", [str(path)]))
    return modules


class build_py(_build_py):
    """Do not ship Python module sources when a compiled extension exists."""

    def find_package_modules(self, package: str, package_dir: str):
        modules = super().find_package_modules(package, package_dir)
        compiled = {ext.name for ext in self.distribution.ext_modules or []}
        return [
            module
            for module in modules
            if f"{module[0]}.{module[1]}" not in compiled
        ]


setup(
    ext_modules=cythonize(
        iter_extension_modules(),
        compiler_directives={"language_level": "3"},
    ),
    cmdclass={"build_py": build_py},
    zip_safe=False,
)
