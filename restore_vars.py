import mip
from typing import Any, List, TypeVar, Tuple, Dict


class RestoreMip:
    """Instantiated with an arbitrary python object containing python-mip
    Variables, the ``restore`` method reconstructs the solution
    of the original object given a list of (var, value) variable solutions.
    """

    def __init__(self, obj):
        self.obj = obj

    @staticmethod
    def var_map(sol: List[Tuple[mip.Var, float]]) -> Dict[int, float]:
        return {id(var): val for var, val in sol}

    def restore(self, sol: List[Tuple[mip.Var, float]]):
        raise NotImplementedError()


class Restore1DList(RestoreMip):
    def __init__(self, obj: List[mip.Var]):
        super().__init__(obj)
        self.ids = [id(x) for x in obj]

    def restore(self, sol) -> List[float]:
        var_map = self.var_map(sol)
        return [var_map.get(i, 0.0) for i in self.ids]


class Restore2DList(RestoreMip):
    def __init__(self, obj: List[List[mip.Var]]):
        super().__init__(obj)
        self.ids = [[id(x) for x in row] for row in obj]

    def restore(self, sol) -> List[List[float]]:
        var_map = self.var_map(sol)
        return [[var_map.get(i, 0.0) for i in row_ids] for row_ids in self.ids]
