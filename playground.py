import GAT
import numpy as np

ref_array = np.random.rand(64,64,64)
eval_array = np.random.rand(64,64,64)

GAT.minimal.gamma(ref_array, eval_array, processor="GPU")