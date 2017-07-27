from server import db, Images
import psutil

import os
import numpy as np
import sys
from scipy.spatial import KDTree

# low mem - 0.18 with this (GiB)
#coords = np.array([img.coord for img in Images.query.yield_per(5).enable_eagerloads(False)])

# high mem - 0.25 with this (GiB)
#coords = np.array([img.coord for img in Images.query.all()])

# adds 2x to memory cost (0.09 to 0.18)
#nn_tree = KDTree(coords)


# current memory use of just this script
pid = os.getpid()
py = psutil.Process(pid)
memUse = py.memory_info()[0]/2.**30 # memuse in GiB (RAM)
print(memUse)
