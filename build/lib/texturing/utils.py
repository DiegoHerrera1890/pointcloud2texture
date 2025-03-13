from pathlib import Path
import numpy as np
from numba import cuda
from scipy import signal
import open3d as o3d
from typing import List, Dict, Tuple
from tqdm import tqdm


def rotation_between_vectors(vectorA: np.ndarray, vectorB: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	'''
	Calculate the ***Rodrigues rotation*** vector(s) and angle(s)

	Perform: VecA_rotated = rotation_matrix ***@*** VecB_normalized

	Parameters:
		vectorA: A vector (d, ) or a matrix (N, d) with N of d dimension vectors
		vectorB: A vector (d, ) or a matrix (N, d) with N of d dimension vectors

	Returns:
		rotation vector(s): A vector (d, ) or a matrix with N of d dimension vectors
		rotation angle(s): A scaler, or a list of scalers

	To convert a rotation vector and angle to **rotation matrix**, use the code snap below.

	```python
	from scipy.spatial.transform import Rotation

	axis_normalized, theta = rotation_between_vectors(vecA, vecB)
	R_target_origin = Rotation.from_rotvec(theta * axis_normalized).as_matrix()
	```
	'''

	if vectorA.shape[0] != vectorB.shape[0]:
		return None

	VecA_normalized = vectorA / np.tile(np.linalg.norm(vectorA, axis=1 if len(vectorA.shape) > 1 else 0), (3, 1)).T
	VecB_normalized = vectorB / np.tile(np.linalg.norm(vectorB, axis=1 if len(vectorB.shape) > 1 else 0), (3, 1)).T
	# Rotation axis
	axis = np.cross(VecB_normalized, VecA_normalized)
	axis_norm = np.linalg.norm(axis, axis=1)
	select = axis_norm > 0
	axis_normalized = np.zeros(axis.shape)
	if not any(select):
		return axis_normalized, np.zeros((axis_normalized.shape[0], ))
	axis_normalized[select] = axis[select] / np.tile(axis_norm[select], (3, 1)).T
	# Rotation theta
	theta = np.arccos(np.sum(np.multiply(VecB_normalized, VecA_normalized), axis=1))
	return (axis_normalized, theta) if len(vectorA.shape) > 1 else (axis_normalized[0], theta[0])


def set_texture_and_mask(uvs: np.ndarray, colors: np.ndarray, texture: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Arguments:
		uvs: N by 2 numpy array
		colors: N by 3 numpy array (same length as the uvs)
		texture: Final output texture (image), build an empty before calling this function
		mask: Final output mask, build an empty before calling this function

	Note: This warper a function, if CUDA is available, it chooses the numba code, otherwise it runs the CPU version. 
	"""
	if cuda.is_available():
		texture_gpu = cuda.to_device(texture)
		mask_gpu = cuda.to_device(mask)
		uvs_gpu = cuda.to_device(uvs)
		colors_gpu = cuda.to_device(colors)
		cuda_threads = 16
		cuda_kernel_shape = ((uvs_gpu.shape[0] + cuda_threads) // cuda_threads, ), (cuda_threads)
		numba_set_texture_and_mask[cuda_kernel_shape](uvs_gpu, colors_gpu, texture_gpu, mask_gpu)
		cuda.synchronize()
		texture = texture_gpu.copy_to_host()
		mask = mask_gpu.copy_to_host()
	else:
		# * Because the projection ray trace is (0, 0, 1), so the (x, y) w.r.t camera is equal to (v, u) w.r.t image
		# ! Note, it's v and u (not u and v)
		for (v, u), color in tqdm(zip(uvs, colors), desc="Backing texture"):
			texture[u, v, :] = (color * 255).astype(np.uint8)
			mask[u, v] = True
	return texture,  mask


@cuda.jit
def numba_set_texture_and_mask(uv: cuda.to_device, color: cuda.to_device, texture: cuda.to_device, mask: cuda.to_device):
	"""
	Arguments:
		uv: The target u and v pixel position of the final output texture and mask
		color: The color which is going to be projected to the uv pixel of the texture
		texture: The final output texture (matrix)
		mask: The final output mask (matrix)
	"""
	uv_idx = cuda.grid(1)
	if uv_idx >= uv.shape[0]:
		return
	# * Because the projection ray trace is (0, 0, 1), so the (x, y) w.r.t camera is equal to (v, u) w.r.t image
	# ! Note, it's v and u (not u and v)
	v, u = uv[uv_idx]
	if v < texture.shape[1] and u < texture.shape[0]:
		texture[u, v, 0] = np.uint8(color[uv_idx, 0] * 255)
		texture[u, v, 1] = np.uint8(color[uv_idx, 1] * 255)
		texture[u, v, 2] = np.uint8(color[uv_idx, 2] * 255)
		mask[u, v] = True


def conv2(origin: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	"""
	A warper function that perform 2D convolution function.

	Arguments:
		origin: M by N matrix
		kernel: Q by R matrix
	
	Return:
		The convolution result
	"""
	if cuda.is_available():
		output = np.empty(origin.shape)
		origin_gpu = cuda.to_device(origin)
		kernel_gpu = cuda.to_device(kernel)
		output_gpu = cuda.to_device(output)
		cuda_threads = 32
		cuda_kernel_shape = ((output.shape[0] + cuda_threads) // cuda_threads, (output.shape[1] + cuda_threads) // cuda_threads), (cuda_threads, cuda_threads)
		numba_conv2[cuda_kernel_shape](origin_gpu, kernel_gpu, output_gpu)
		cuda.synchronize()
		output = output_gpu.copy_to_host()
	else:
		output = signal.convolve2d(origin, kernel, mode='same')
	return output


@cuda.jit
def numba_conv2(origin: cuda.to_device, kernel: cuda.to_device, output: cuda.to_device):
	"""
	Arguments:
		origin: The origin 2D matrix (in our case, it's a distance weighting matrix)
		kernel: The kernel, that will be used for convolution
		output: For now, the size of the "output" aligns to the origin
	"""
	x, y = cuda.grid(2)
	if x >= origin.shape[0] or y >= origin.shape[1]:
		return
	k_size = kernel.shape[:2]
	o_size = origin.shape[:2]
	k_offset = ( int(kernel.shape[0] // 2), int(kernel.shape[1] // 2) )
	value = 0
	for k_idx_x in range(k_size[0]):
		for k_idx_y in range(k_size[1]):
			o_idx_x = x - k_offset[0] + k_idx_x
			o_idx_y = y - k_offset[1] + k_idx_y
			if o_idx_x < 0 or o_idx_y < 0 or o_idx_x >= o_size[0] or o_idx_y >= o_size[1]:
				continue
			value += kernel[k_idx_x, k_idx_y] * origin[o_idx_x, o_idx_y]
	output[x, y] = value


def conv2_rgb_origin(origin: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	"""
	Arguments:
		origin: M * N * 3 matrix, in this case, it's a RGB image
		kernel: P * Q matrix, plays as a kernel
	"""
	if cuda.is_available():
		output = np.empty(origin.shape)
		origin_gpu = cuda.to_device(origin)
		kernel_gpu = cuda.to_device(kernel)
		output_gpu = cuda.to_device(output)
		cuda_threads = 32
		cuda_kernel_shape = ((origin.shape[0] + cuda_threads) // cuda_threads, (origin.shape[1] + cuda_threads) // cuda_threads, 3), (cuda_threads, cuda_threads)
		numba_conv2_rgb_origin[cuda_kernel_shape](origin_gpu, kernel_gpu, output_gpu)
		cuda.synchronize()
		output = output_gpu.copy_to_host()
	else:
		output = np.empty(origin.shape)
		output[:, :, 0] = signal.convolve2d(origin[:, :, 0], kernel, mode='same')
		output[:, :, 1] = signal.convolve2d(origin[:, :, 1], kernel, mode='same')
		output[:, :, 2] = signal.convolve2d(origin[:, :, 2], kernel, mode='same')
	return output


@cuda.jit
def numba_conv2_rgb_origin(origin: cuda.to_device, kernel: cuda.to_device, output: cuda.to_device):
	"""
	Arguments:
		origin: A matrix which has dimension M * N * 3 (a.k.a image)
		kernel: A P * Q matrix (in our case, it's distance weighting matrix)
		output: The size of this output aligns to the origin size
	"""
	x, y, z = cuda.grid(3)
	if x >= origin.shape[0] or y >= origin.shape[1] or z >= 3:
		return
	k_size = kernel.shape[:2]
	o_size = origin.shape[:2]
	k_offset = ( int(kernel.shape[0] // 2), int(kernel.shape[1] // 2) )
	value = 0
	for k_idx_x in range(k_size[0]):
		for k_idx_y in range(k_size[1]):
			o_idx_x = x - k_offset[0] + k_idx_x
			o_idx_y = y - k_offset[1] + k_idx_y
			if o_idx_x < 0 or o_idx_y < 0 or o_idx_x >= o_size[0] or o_idx_y >= o_size[1]:
				continue
			value += kernel[k_idx_x, k_idx_y] * origin[o_idx_x, o_idx_y, z]
	output[x, y, z] = value


def validate_ply_file(file_path, file_type):
    """ Validates that the file exists, has a .ply extension, and is not empty. """
    path = Path(file_path)

    # Check if the file exists
    if not path.exists():
        raise ValueError(f"Error: The {file_type} file '{file_path}' does not exist.")

    # Check if the file has a .ply extension
    if path.suffix.lower() != ".ply":
        raise ValueError(f"Error: The {file_type} file '{file_path}' is not a .ply file.")

    # Check if the PLY file is empty (zero points)
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"Error: The {file_type} file '{file_path}' is empty.")

    return path



