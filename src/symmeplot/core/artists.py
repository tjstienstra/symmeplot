"""Definition of the base class for the artists."""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["ArtistBase"]


class ArtistBase(ABC):
    """Base class for the artists."""

    @abstractmethod
    def update_data(self, *args: object) -> None:
        """Update the data of the artist."""
