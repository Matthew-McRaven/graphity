import numpy as np
import torch

def convert_np_torch(item):
	if item in convert_np_torch._info: return convert_np_torch._info[item]
	elif item == np.float32: return torch.float32
	elif item == np.int8: return torch.int8
	print(f"Dtype ({item}) not implemented.")
	assert 0 and "Dtype not implemented."

convert_np_torch._info = {
		np.bool       : torch.bool,
		np.uint8      : torch.uint8,
		np.int8       : torch.int8,
		np.int16      : torch.int16,
		np.int32      : torch.int32,
		np.int64      : torch.int64,
		np.float16    : torch.float16,
		np.float32    : torch.float32,
		np.float64    : torch.float64,
		np.complex64  : torch.complex64,
		np.complex128 : torch.complex128}

