"""Utilities for testing."""

from __future__ import annotations

import os

ON_CI = os.getenv("CI", None) == "true"
