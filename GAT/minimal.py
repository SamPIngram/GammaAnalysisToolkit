import cupy as cp
import numpy as np
import time    

def initalise(ref_array, eval_array, processor):
    """Checks data and moves to processor if required.

    Args:
        ref_array (_type_): _description_
        eval_array (_type_): _description_
        processor (_type_): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: Checked arrays on correct processor.
    """
    assert ref_array.shape == eval_array.shape
    if processor != "GPU" and processor != "CPU":
        raise Exception(f'processor can only be "CPU" or "GPU" got "{processor}"')
    if processor=="GPU" and (type(ref_array) == np.ndarray or type(eval_array) == np.ndarray):
        print("Converting to GPU")
        ref_array = cp.array(ref_array, dtype=cp.float32)
        eval_array = cp.array(eval_array, dtype=cp.float32)

    return ref_array, eval_array

@cp.fuse(kernel_name='elementwise_gamma')
def elementwise_gamma(indicies: cp.ndarray, ref_array: cp.ndarray, eval_array: cp.ndarray, shape: cp.array):
    i = int(indicies % shape[0])
    j = int(indicies % (shape[0]*shape[1]))
    k = int(indicies % (shape[0]*shape[1]*shape[2]))
    ref_val = ref_array[i][j][k]
    eval_vals = cp.array([eval_array[i][j][k],eval_array[i+1][j][k],eval_array[i-1][j][k],eval_array[i][j+1][k]],eval_array[i][j-1][k])
    diff = cp.absolute(cp.subtract(eval_vals, ref_val))
    return cp.min(diff)

def gamma(ref_array, eval_array, threshold_perc=10, processor="GPU"):
    ref_array, eval_array = initalise(ref_array, eval_array, processor)
    indicies = cp.arange(0, ref_array.flatten().shape[0], 1, dtype=cp.int8)
    res = elementwise_gamma(indicies, ref_array, eval_array, cp.array(ref_array.shape))
    print(res)
    