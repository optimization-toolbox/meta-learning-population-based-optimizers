
from .base import BaseGA
from .mutation import RealMutator
from .selection import TruncatedSelection


class TruncatedRealMutatorGA_Table(TruncatedSelection, RealMutator, BaseGA):
    pass
