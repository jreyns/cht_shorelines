from typing import List, Union
import numpy as np
from pydantic import BaseModel, field_validator

class GridSpec(BaseModel):
    """
    Represents multiple curvi-linear or linear grids.

    Option 1: [[x1, y1, x2, y2], ...]
    Option 2: [np.ndarray(Nx2), np.ndarray(Mx2), ...]
    """

    grids: Union[List[List[float]], List[np.ndarray]]

    @field_validator("grids")
    def check_format(cls, v):
        # Option 1 — linear grids (list of [x1,y1,x2,y2])
        if all(isinstance(g, list) and len(g) == 4 and all(isinstance(x, (int, float)) for x in g) for g in v):
            return v
        
        # Option 2 — curvilinear grids (list of Nx2 numpy arrays)
        elif all(isinstance(g, np.ndarray) and g.ndim == 2 and g.shape[1] == 2 for g in v):
            return v
        
        raise ValueError(
            "Each grid must be either a [x1, y1, x2, y2] list "
            "or an Nx2 numpy array of coordinates."
        )
# Use:    
# from gridspec import GridSpec

# linear_grids = [
#     [0.0, 0.0, 10.0, 0.0],
#     [0.0, 0.0, 0.0, 10.0],
# ]

# g = GridSpec(grids=linear_grids)
# print("Validated type:", type(g.grids))
# print("Success:", g)