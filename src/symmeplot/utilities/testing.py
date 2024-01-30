import os

ON_CI = os.getenv("CI", None) == "true"
