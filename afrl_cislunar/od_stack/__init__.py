"""Cislunar orbit determination stack: observations -> IOD -> propagation -> results."""

# Load the IOD registry first: od_stack.iod registers the Part B IOD modules from its last lines, and
# those modules import IODFailure from it. Importing one of them first (e.g. od_stack.iod_laplace)
# would otherwise reach that registration while it is only half loaded.
from . import iod  # noqa: E402,F401

