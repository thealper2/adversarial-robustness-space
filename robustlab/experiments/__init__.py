"""Importing this package registers every experiment in ``robustlab.core.REGISTRY``."""

from .. import compat  # noqa: F401  (must run before the ART attacks are used)
from . import (  # noqa: F401
    baddet,
    certification,
    defence_pre,
    defence_train,
    evasion_black,
    evasion_classic,
    evasion_white,
    extraction,
    inference,
    poison_detection,
    poisoning,
    universal,
)
