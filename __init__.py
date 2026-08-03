"""meridian_simulator – Simulate Meridian-compatible MMM datasets.

Public API::

    from meridian_simulator import MeridianSimulator
    from meridian_simulator.config import (
        SimulationConfig,
        MediaChannelConfig,
        RFChannelConfig,
        OrganicMediaChannelConfig,
        OrganicRFChannelConfig,
        NonMediaChannelConfig,
        ContextVariableConfig,
        BaselineConfig,
        SeasonalityComponent,
    )

Note: ``MeridianSimulator`` and ``SimulationResult`` are imported lazily to
avoid loading TensorFlow at import time of the config submodule.
"""

from meridian_simulator.config import (
    BaselineConfig,
    CollinearVariableConfig,
    ContextVariableConfig,
    EndogenousVariableConfig,
    MediaChannelConfig,
    NonMediaChannelConfig,
    OrganicMediaChannelConfig,
    OrganicRFChannelConfig,
    PromoEventConfig,
    RFChannelConfig,
    SeasonalityComponent,
    SimulationConfig,
)


__version__ = "2.1.0"


def __getattr__(name):
    if name in ("MeridianSimulator", "SimulationResult"):
        from meridian_simulator.simulator import MeridianSimulator, SimulationResult  # noqa: F401

        globals()["MeridianSimulator"] = MeridianSimulator
        globals()["SimulationResult"] = SimulationResult
        return globals()[name]
    raise AttributeError(f"module 'meridian_simulator' has no attribute {name!r}")


__all__ = [
    "MeridianSimulator",
    "SimulationResult",
    "SimulationConfig",
    "MediaChannelConfig",
    "RFChannelConfig",
    "OrganicMediaChannelConfig",
    "OrganicRFChannelConfig",
    "NonMediaChannelConfig",
    "ContextVariableConfig",
    "CollinearVariableConfig",
    "EndogenousVariableConfig",
    "PromoEventConfig",
    "BaselineConfig",
    "SeasonalityComponent",
]
