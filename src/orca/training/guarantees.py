"""Physical properties a surrogate can enforce by construction.

Both halves of the output path can make such a promise: the architecture (a
stable-by-parameterization pole, a symmetric residue matrix) and the output
representation (an upper-triangle codec cannot express a non-reciprocal
S-matrix). They live here rather than in either module so that
:mod:`orca.training.codecs` and :mod:`orca.training.models.base_model` can both
declare them without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields


@dataclass(frozen=True)
class PhysicsGuarantees:
    """Physical properties a model or codec enforces *by construction*.

    These are guarantees, not aspirations: set a flag only if the architecture
    or the output representation makes violations impossible (a structurally
    symmetric output, a stable-by-parameterization pole, ...), not if the
    network merely tends to learn it.

    The guarantees are written into the exported ONNX metadata so downstream
    consumers (COBRA) know what they are holding, and let the trainer skip
    penalty terms that would be redundant.

    Attributes:
        passive: sigma_max(S) <= 1 for all frequencies.
        reciprocal: S == S.T.
        causal: The response is causal (e.g. a rational/pole-residue head).
        stable: All poles lie in the left half plane.
    """

    passive: bool = False
    reciprocal: bool = False
    causal: bool = False
    stable: bool = False

    def __or__(self, other: "PhysicsGuarantees") -> "PhysicsGuarantees":
        """Combine two sets of guarantees, keeping every property either one promises.

        Used to merge what the architecture guarantees with what the output
        codec guarantees, since a property enforced by either is enforced.
        """
        if not isinstance(other, PhysicsGuarantees):
            return NotImplemented
        return PhysicsGuarantees(
            **{f.name: getattr(self, f.name) or getattr(other, f.name) for f in fields(self)}
        )

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)
