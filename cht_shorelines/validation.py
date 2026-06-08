from typing import List

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class GridSpec(BaseModel):
    """
    Represents multiple curvi-linear grids.

    Each grid must be provided as an ``Nx2`` numpy array.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grids: List[np.ndarray]

    @field_validator("grids")
    def check_format(cls, v: List[np.ndarray]) -> List[np.ndarray]:
        """
        Validate that each grid is an ``Nx2`` numpy array.

        Parameters
        ----------
        v : list of numpy.ndarray
            Candidate grid arrays.

        Returns
        -------
        list of numpy.ndarray
            Validated grid arrays.
        """
        if not all(
            isinstance(g, np.ndarray) and g.ndim == 2 and g.shape[1] == 2
            for g in v
        ):
            raise ValueError("Each grid must be an Nx2 numpy array of coordinates.")
        return v
