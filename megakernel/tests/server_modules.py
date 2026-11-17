"""The server's modules as one object for the tests: mk.NAME reads NAME from whichever module has it, and setting it
(mock.patch.object) replaces it in every module that holds it, because `from x import NAME` gives each its own binding."""
import importlib
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent / "server"


class ServerModules:
    def __init__(self, modules):
        vars(self).update(_modules=modules, _saved={})

    def __getattr__(self, name):
        for m in self._modules:
            if name in vars(m):
                return vars(m)[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        holders = [m for m in self._modules if name in vars(m)]
        if not holders:
            raise AttributeError(name)
        self._saved.setdefault(name, []).append([(m, vars(m)[name]) for m in holders])
        for m in holders:
            setattr(m, name, value)

    def __delattr__(self, name):  # (how mock.patch undoes a patch of a name outside the target's own __dict__)
        for m, old in self._saved[name].pop():
            setattr(m, name, old)


def load():
    """a fresh copy of every server module, so that module state (main() sets some) does not cross test classes"""
    for name in [n for n in sys.modules if n.startswith("mk_")]:
        del sys.modules[name]
    if str(SERVER) not in sys.path:
        sys.path.insert(0, str(SERVER))
    importlib.import_module("mk_server")
    return ServerModules([m for n, m in sys.modules.items() if n.startswith("mk_")])
