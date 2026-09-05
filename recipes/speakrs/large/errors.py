"""Typed failures for the Large recipe command surface."""

from __future__ import annotations


class LargeError(RuntimeError):
    """A machine-readable Large recipe failure."""

    def __init__(self, code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_json(self) -> dict[str, object]:
        """Return the JSON error body written on nonzero exits."""

        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }


class ContractError(LargeError):
    """A schema, identity, or state-machine violation."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__("contract", message, details)


class PreparationError(LargeError):
    """A data-release construction or verification failure."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__("preparation", message, details)


class RuntimeGateError(LargeError):
    """A training, recovery, budget, or selection gate failure."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__("runtime", message, details)
