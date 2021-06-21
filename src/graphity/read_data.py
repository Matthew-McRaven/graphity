import torch

def read_betre_format(filename):
	"""
	Read a file stored in the format of a file in data/ground.
	These files came from Betre in Dec 2020, and represent spherical/geometric graphs of various sizes.
	"""
	ret_tens = []
	with open(filename) as  data_file:
		for line in data_file.readlines():
			nums = line.translate({ord(i): None for i in "\" {}\n"}).split(",")
			nums = [int(i) for i in nums]
			dim = int(len(nums) **.5)
			nums = torch.Tensor(nums).view((dim,dim))
			ret_tens.append(nums)
	return ret_tens