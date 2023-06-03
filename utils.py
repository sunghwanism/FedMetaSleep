import os

import numpy as np
import pandas as pd



def count_stage(total_stage):
    subj_stage = []
    for i in range(len(total_stage)):
        wake = (total_stage[i]==0).sum()
        n1 = (total_stage[i]==1).sum()
        n2 = (total_stage[i]==2).sum()
        n3 = (total_stage[i]==3).sum()
        rem = (total_stage[i]==5).sum()
        
        subj_stage.append([wake, n1, n2, n3, rem])
    return np.array(subj_stage)