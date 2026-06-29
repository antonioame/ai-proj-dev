"""Central driver registry — single place to resolve driver names to instances."""

from __future__ import annotations

from drivers.base_driver import BaseDriver

_AVAILABLE = ("rule_based", "optimal", "bc")


def load_driver(name: str) -> BaseDriver:
    """Return an instance of the named driver.

    Parameters
    ----------
    name:
        ``"rule_based"`` – physics-based baseline (stable, ~148 s lap).
        ``"optimal"``    – trajectory-follower with late braking (in progress).
        ``"bc"``         – behavioral cloning from rule-based demonstrations.

    Raises
    ------
    ValueError
        If *name* is not recognised.
    """
    if name == "rule_based":
        from drivers.rule_based.driver import RuleBasedDriver
        return RuleBasedDriver()
    if name == "optimal":
        from drivers.optimal.driver import OptimalLineDriver
        return OptimalLineDriver()
    if name == "bc":
        from drivers.bc.driver import BCDriver
        return BCDriver()
    raise ValueError(
        f"Unknown driver '{name}'. Available: {', '.join(_AVAILABLE)}"
    )
