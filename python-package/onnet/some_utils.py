import numpy as np

def split__sections(dim_0,nClass):
    split_dim = range(dim_0)
    sections=[]
    for arr in np.array_split(np.array(split_dim), nClass):
        sections.append(len(arr))
    return sections