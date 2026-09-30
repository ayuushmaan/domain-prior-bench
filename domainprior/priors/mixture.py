"""Alpha-mixture of domain + generic samplers (+ optional curriculum)."""
from __future__ import annotations
import numpy as np
from .base import Task
from .credit import sample_credit_task
from .generic import sample_generic_task


def sample_mixture(rng, alpha, generic_sampler=sample_generic_task, **kw) -> Task:
    """alpha = share of domain tasks. Extra kw goes to the domain sampler."""
    if rng.random() < alpha:
        t = sample_credit_task(rng, **kw)
        t.meta["mixture"] = "domain"
        return t
    t = generic_sampler(rng)
    t.meta["mixture"] = "generic"
    return t
