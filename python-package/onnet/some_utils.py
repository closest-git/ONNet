import numpy as np
import math

def split__sections(dim_0,nClass):
    split_dim = range(dim_0)
    sections=[]
    for arr in np.array_split(np.array(split_dim), nClass):
        sections.append(len(arr))
    assert len(sections) > 0
    return sections

def shrink(x0,x1,max_sz=2):
    if x1-x0>max_sz:
        center=(x1+x0)//2
        #x1 = x0+max_sz
        x1 = center + max_sz // 2
        x0 = center - max_sz // 2
    return x0,x1

def split_regions_2d(shape,nClass):
    dim_1,dim_2=shape[-1],shape[-2]
    n1 = (int)(math.sqrt(nClass))
    n2 = (int)(math.ceil(nClass/n1))
    assert n1*n2>=nClass
    section_1 = split__sections(dim_1, n1)
    section_2 = split__sections(dim_2, n2)
    regions = []
    x1,x2=0,0
    for sec_1 in section_1:
        for sec_2 in section_2:
            #box=(x1,x1+sec_1,x2,x2+sec_2)
            box = shrink(x1,x1+sec_1)+shrink(x2,x2+sec_2)
            regions.append(box)
            if len(regions)>=nClass:
                break
            x2 = x2 + sec_2
        x1 = x1 + sec_1;    x2=0
    return regions