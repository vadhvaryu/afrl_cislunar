"""Start the Java VM for Orekit once, and load orekit-data.

The data folder comes from config.yaml (orekit.data_path) or, if that is empty,
the OREKIT_DATA environment variable. The Batch Least Squares stage needs it
(leap seconds for the observer station); the IOD and propagators do not.
"""

from __future__ import annotations

import os

_started = False


def start_orekit(data_path: str | None = None) -> None:
    global _started
    if _started:
        return
    import orekit_jpype

    orekit_jpype.initVM()
    data_path = data_path or os.environ.get("OREKIT_DATA")
    if data_path:
        data_path = os.path.expanduser(data_path)
        from orekit_jpype.pyhelpers import setup_orekit_data

        setup_orekit_data(filenames=data_path, from_pip_library=False)
    _started = True
