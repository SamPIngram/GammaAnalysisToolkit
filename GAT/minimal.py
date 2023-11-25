import cupy as cp
import numpy as np
import time

def gamma(ref_array, eval_array, processor="GPU"):
    if processor != "GPU" and processor != "CPU":
        raise Exception(f'processor can only be "CPU" or "GPU" got "{processor}"')
    if processor=="GPU" and (type(ref_array) == np.ndarray or type(eval_array) == np.ndarray):
        print("Converting to GPU")
        ref_array = cp.array(ref_array)
        eval_array = cp.array(eval_array)
    