from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["ArtistBase"]


class ArtistBase(ABC):
    """Base class for the artists."""

    @abstractmethod
    def update_data(self, *args):
        """Update the data of the artist."""
