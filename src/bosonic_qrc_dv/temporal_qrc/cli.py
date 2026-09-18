"""Guard against mislabeling static QRP as temporal recurrence."""

from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "Temporal DV IPC is unavailable: no validated recurrent Perceval "
        "quantum channel is implemented. Use static-capacity instead."
    )
