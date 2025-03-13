import argparse
from pathlib import Path
import copy
import os, sys
import datetime
from time import sleep

from matplotlib import pyplot as plt
import cv2
import glog
from tqdm import tqdm
from typing import List, Dict, Tuple
from dataclasses import dataclass
# from src.inpainting.ddnm_inpainting import Inpainter
# import torch
# from numba import cuda
import open3d as o3d
import numpy as np
import rpack
# import torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from utils import rotation_between_vectors
from utils import set_texture_and_mask, conv2, conv2_rgb_origin

@dataclass
class rect_info:
	origin: Tuple[float, float]
	length: Tuple[float, float]

	def get_origin_in_pixel(self, resolution: int, axis_order: Tuple[int, int]=(0, 1)) -> Tuple[int, int]:
		# TBC: Why not just output as a numpy array?
		return (int(self.origin[axis_order[0]] * resolution), int(self.origin[axis_order[1]] * resolution))

	def get_length_in_pixel(self, resolution: int, axis_order: Tuple[int, int]=(0, 1)) -> Tuple[int, int]:
		return (int(self.length[axis_order[0]] * resolution) + 1, int(self.length[axis_order[1]] * resolution) + 1)

	def apply_scale(self, scale_factor: float=1.0):
		assert(scale_factor > 0.0)
		self.origin = (self.origin[0] * scale_factor, self.origin[1] * scale_factor)
		self.length = (self.length[0] * scale_factor, self.length[1] * scale_factor)


@dataclass
class cluster_info:
	id: int
	triangle_selects: List[ np.ndarray ]
	voxel_rect_info: rect_info
	rect_texture: np.ndarray
	rect_mask: np.ndarray

	def __iter__(self):
		yield self.id
		yield self.triangle_selects
		yield self.voxel_rect_info
		yield self.rect_texture
		yield self.rect_mask


class text_recon():
	'''
	This is a class of "six_projection_planes" module by using six projection planes approach
	'''
	default_config = {
		"voxel_size": 0.2,
		"align_the_axis": False,
		"default_color": [51, 51, 204],
		"texture_resolution": 2048,
		"texture_refinement_mask_size": 0,
		"single_projection_plane": False,
		"clustering_meshes": False,
	}

	default_target_planes = [
		(np.array([0, 0, 1]), "z_up"), 
		(np.array([1, 0, 0]), "x_right"), 
		(np.array([-1, 0, 0]), "x_left"), 
		(np.array([0, 1, 0]), "y_forward"), 
		(np.array([0, -1, 0]), "y_backward"), 
		(np.array([0, 0, -1]), "z_down"), # Collect everything, since it's the last one
	]

	def __init__(self, source_mesh_path: Path, reference_pcd_path: Path, output_path: Path=None, config: Dict=None, output_debug_files: bool=False):
		"""
		Arguments:
			source_mesh_path: the input source mesh file, will refer to it's vertices and triangulates
			reference_pcd_path: the input colored point cloud will refer to it's points and color information
			output_path: a parent folder for storing the output results / files, it will create a child folder
			config: a json stream (not a path) which stores the parameters for the algorithm
			output_debug_files: set to "True" if you are debugging and want to see the details of the "six planes" results
		"""
		self.output_debug_files = output_debug_files
		self.config = config if config is not None else self.default_config
		self.source_mesh_path = source_mesh_path
		self.reference_pcd_path = reference_pcd_path
		self.output_path = output_path if output_path is not None else Path("outputs/spp_text_recon_debug") if self.output_debug_files else Path("outputs")
		# * timestamp as sub folder name
		# self.output_path = self.output_path / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		self.output_path = self.output_path / datetime.datetime.now().strftime("%Y%m%d")
		self.output_dir = self.output_path / source_mesh_path.parent.stem
		self.output_dir.mkdir(parents=True, exist_ok=True)
		# self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# self.inpainter = Inpainter(self.device)
		glog.info(f"SPP is running with config = {self.config}")
		glog.info(f"{self.output_debug_files = }")
		glog.info(f"{self.source_mesh_path = }")
		glog.info(f"{self.reference_pcd_path = }")
		glog.info(f"The output files will store on {self.output_dir}")
		glog.info(f"Initialization done, please call run() to process the remain procedures.")

	def auto_parameterization(self):
		"""
		Check if there is any non reasonable parameters, and estimate the reasonable parameters
		"""
		if self.config["texture_refinement_mask_size"] <= 0:
			# From experience (for now), 15 meters need 50 pixel kernel for 2048 pixel resolution texture
			#     15                            50 * 2048
			# ---------- = -------------------------------------------------
			# z-distance   texture_refinement_mask_size * texture_resolution
			z_distance = np.min(self.source_mesh.get_max_bound()[2] - self.source_mesh.get_min_bound()[2])
			texture_resolution = self.config["texture_resolution"]
			max_kernel_size = texture_resolution // 16
			self.config["texture_refinement_mask_size"] = min(int((50 * 2048 * z_distance) / (texture_resolution * 15)), max_kernel_size)
		glog.info(f"After auto parameterization, {self.config = }")

	def plane_vector_to_projection_matrix(self, target_plane_vector: np.ndarray) -> np.ndarray:
		"""
		A sub function for converting plane vector to (orthophoto) projection matrix

		Arguments:
			target_plane_vector: A plane vector (e.g. for Z-axis from top --> (0, 0, 1))

		Returns:
			A orthophoto projection matrix
		"""
		
		projection_matrix = np.zeros((3, 3))
		if any(target_plane_vector > 0):
			xyz = np.where(target_plane_vector > 0.5)[0]
			assert(len(xyz) == 1)
			if xyz == 0:
				projection_matrix = [
					[0, 1, 0],
					[0, 0, 1],
					[1, 0, 0],
				]
			elif xyz == 1:
				projection_matrix = [
					[0, 0, 1],
					[1, 0, 0],
					[0, 1, 0],
				]
			else:
				projection_matrix = [
					[1, 0, 0],
					[0, 1, 0],
					[0, 0, 1],
				]
		else:
			xyz = np.where(target_plane_vector < 0)[0]
			assert(len(xyz) == 1)
			if xyz == 0:
				
				projection_matrix = [
					[0, 0, 1],
					[0, 1, 0],
					[-1, 0, 0],
				]
			elif xyz == 1:
				projection_matrix = [
					[0, -1, 0],
					[1, 0, 0],
					[0, 0, 1],
				]
			else:
				projection_matrix = [
					[1, 0, 0],
					[0, -1, 0],
					[0, 0, -1],
				]

		glog.debug(f"{projection_matrix = }")
		transform_projection_matrix = np.matrix(np.identity(4))
		transform_projection_matrix[:3, :3] = np.matrix(projection_matrix)
		return transform_projection_matrix

	def refine_texture(self, alts_name, texture: o3d.geometry.Image, mask: np.ndarray, kernel_size: int=11) -> o3d.geometry.Image:
			"""
			A member function for blending the pixel (/texel) by convoluting texture by using distance as weight kernel

			Arguments:
				texture: A 2D color image data (M * N * 3) in numpy array format
				mask: A 2D mask in bool format which indicates which pixels are from point cloud (/important)
				kernel_size: When fuse / blend the color, the largest pixel distance we are going to take 
			"""
			glog.info(f"The kernel size is: {kernel_size}")

			# Create the kernel --> distance matrix as weight matrix
			dist_array = np.square(np.arange(kernel_size) - np.floor(kernel_size * 0.5))
			dist_matrix_kernel = np.sqrt(dist_array[:, None] + dist_array[None, :])

			sub_texture = np.asarray(texture)
			sub_mask = mask.astype(bool)

			divider = conv2(sub_mask, dist_matrix_kernel)

			# Avoid division by zero
			unprocessed_mask = divider <= 0
			# divider[divider] = 1.0
			divider[divider <= 0] = 1.0 
			masked_texture = sub_texture.copy().astype(np.float64)
			masked_texture[~sub_mask] = 0

			blended_texture = conv2_rgb_origin(masked_texture, dist_matrix_kernel)
			blended_texture = np.clip(blended_texture * (~sub_mask[:, :, np.newaxis]) / divider[:, :, np.newaxis], 0, 255).astype(np.uint8)


			# # Convert to grayscale for mask generation
			gray_mask = cv2.cvtColor(blended_texture, cv2.COLOR_RGB2GRAY)

			# # Create binary mask for inpainting (0 = keep, 255 = fill)
			threshold_value = 50  # Adjust as needed
			inpainting_mask = (gray_mask < threshold_value).astype(np.uint8) * 255
			inpainting_mask = cv2.bitwise_not(inpainting_mask)  # Ensure correct mask inversion

			

			# Save mask for debugging
			# mask_path = os.path.join("/home/diego/pointcloud2texture/outputs", f"mask_for_inpainting_{alts_name}.jpg")
			# cv2.imwrite(mask_path, inpainting_mask)

			# Load input texture
			# image_path = os.path.join("/home/diego/pointcloud2texture/outputs/textures", f"original_texture_{alts_name}.png")

			# if not os.path.exists(image_path):
			# 	glog.error(f"ERROR: Image file not found: {image_path}")
			# 	return texture
			# if not os.path.exists(mask_path):
			# 	glog.error(f"ERROR: Mask file not found: {mask_path}")
			# 	return texture

			image = cv2.cvtColor(np.asanyarray(texture), cv2.COLOR_RGB2BGR)
			# image = cv2.imread(image_path, cv2.IMREAD_COLOR)
			# mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

			# Apply inpainting
			# inpainted = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_NS  )
			inpainted = cv2.inpaint(image, inpainting_mask, inpaintRadius=5, flags=cv2.INPAINT_NS  )


			# Save inpainted texture
			# output_inpainted_path = os.path.join("/home/diego/pointcloud2texture/outputs/inpainted", f"inpainted_{alts_name}.png")
			# cv2.imwrite(output_inpainted_path, inpainted)

			refined_sub_texture1_inpainted = inpainted
			refined_sub_texture1_inpainted[unprocessed_mask, :] = sub_texture[unprocessed_mask, :]

			

			return o3d.geometry.Image(refined_sub_texture1_inpainted)

	
	
	# def project_to_plane(self, target_plane_vector: np.ndarray, sub_mesh: o3d.geometry.TriangleMesh, sub_pcd: o3d.geometry.PointCloud, texture_resolution: int=2048, normalize_scale_factor: float=None) -> Tuple[np.ndarray, o3d.geometry.Image, np.ndarray]:
	def project_to_plane(self, target_plane_vector: np.ndarray, alts_name: str, sub_mesh: o3d.geometry.TriangleMesh, sub_pcd: o3d.geometry.PointCloud, texture_resolution: int=2048, normalize_scale_factor: float=None) -> Tuple[np.ndarray, o3d.geometry.Image, np.ndarray]:

		"""
		Project mesh faces vertices onto the target plane --> uv table
		Project point cloud onto the target plane --> texture image

		Arguments:
			target_plane_vector: plane's normal vector, e.g. from top view, [0, 0, 1]
			sub_mesh: a sub set of mesh, we use this to filter the point cloud
			sub_pcd: a sub set of colored point cloud that we are going to project onto the "plane"
			texture_resolution: the target texture resolution
			normalize_scale_factor: use the "scale_factor" to normalize the uv table.
		"""
		transform_projection_matrix = self.plane_vector_to_projection_matrix(target_plane_vector)

		# Create the uv table
		glog.info("Building the uv table ... ")
		# ! The size is (face number * 3) by 2
		temp_sub_mesh = copy.deepcopy(sub_mesh)
		temp_sub_mesh.transform(transform_projection_matrix)
		uv_table_vertex_ref = np.array(temp_sub_mesh.vertices)[:, :2]
		uv_table = np.zeros((len(temp_sub_mesh.triangles) * 3, 2))
		for face_idx, face in tqdm(enumerate(temp_sub_mesh.triangles), desc="Building vu table"):
			uv_table[3 * face_idx : 3 * (face_idx + 1), :] = [ uv_table_vertex_ref[vex_idx] for vex_idx in face ]
		glog.info("Built the uv table")

		glog.info("Transforming the point cloud w.r.t camera coordinate")
		# ! Don't need this, unless you are debugging (for speeding up)
		temp_sub_pcd = copy.deepcopy(sub_pcd) if self.output_debug_files else sub_pcd
		temp_sub_pcd.transform(transform_projection_matrix)
		# TBC: Blending image pixels by voxelization?
		uvs = np.array(temp_sub_pcd.points)[:, :2]
		colors = temp_sub_pcd.colors
		glog.info("Transformed the point cloud w.r.t camera coordinate")

		# ! Don't forget to do the normalization
		dist_min = np.min([np.min(temp_sub_mesh.vertices, axis=0), np.min(temp_sub_pcd.points, axis=0)], axis=0)[:2]
		dist_max = np.max([np.max(temp_sub_mesh.vertices, axis=0), np.max(temp_sub_pcd.points, axis=0)], axis=0)[:2]
		dist_max_min = (dist_max - dist_min)
		dist_max_min = np.max(dist_max_min)
		# If we have the "normalize_scale_factor" from caller
		dist_max_min = np.max(normalize_scale_factor) if normalize_scale_factor is not None else dist_max_min
		glog.debug(f"Using [{dist_max_min}] to normalize the uv table")
		uv_table = (uv_table - dist_min) / dist_max_min
		uvs = (texture_resolution - 1) * (uvs - dist_min) / dist_max_min
		uvs = uvs.astype(np.int64)
		
		glog.info("Starting baking the texture ...")
		texture = np.zeros((texture_resolution, texture_resolution, 3), dtype=np.uint8)
		mask = np.zeros((texture_resolution, texture_resolution), dtype=bool)
		# * Set the default color to [0.8, 0.2, 0.2]
		texture[:] = self.config["default_color"]
		# * Project to the plane
		texture, mask = set_texture_and_mask(uvs, colors, texture, mask)
		glog.info("Texture baked")

		texture = o3d.geometry.Image(texture)
		
		
		# output_dir = "outputs/textures"
		# os.makedirs(output_dir, exist_ok=True)
		# glog.info(f"Saving original Texturing at {output_dir}")
		# texture_np = np.asanyarray(texture)
		# output_path = os.path.join(output_dir, f"original_texture_{alts_name}.png")
		# cv2.imwrite(output_path, cv2.cvtColor(texture_np, cv2.COLOR_RGB2BGR))
		# glog.info(f"Original Texture saved as {output_path}")

		# Convert boolean mask to an RGB image
		mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

		# Set masked areas to a specific color (e.g., red)
		mask_rgb[mask] = [255, 0, 0]  # Red areas indicate missing data

		# Set output path for the mask image
		# output_dir = "outputs/masks"
		# os.makedirs(output_dir, exist_ok=True)
		# mask_output_path = os.path.join(output_dir, f"original_mask_{alts_name}.png")

		# Save the mask image
		# cv2.imwrite(mask_output_path, mask_rgb)

		# glog.info(f"Original Mask saved as {mask_output_path}")


		return uv_table, texture, mask

	def project_mesh_cluster_to_plane(self, target_plane_vector: np.ndarray, sub_mesh: o3d.geometry.TriangleMesh, sub_pcd: o3d.geometry.PointCloud, texture_resolution: int=2048, normalize_scale_factor: float=None) -> Tuple[np.ndarray, o3d.geometry.Image, np.ndarray]:
		"""
		Arguments:
			target_plane_vector:
		"""
		
		transform_projection_matrix = self.plane_vector_to_projection_matrix(target_plane_vector)

		# ! The size is (face number * 3) by 2
		temp_sub_mesh = copy.deepcopy(sub_mesh)
		temp_sub_mesh.transform(transform_projection_matrix)
		temp_sub_pcd = copy.deepcopy(sub_pcd)
		temp_sub_pcd.transform(transform_projection_matrix)

		uv_table_vertex_ref = np.array(temp_sub_mesh.vertices)[:, :2]
		uv_table = np.zeros([len(temp_sub_mesh.triangles) * 3, 2])
		# # (N, 3) matrix to (N * 3, ) array
		uv_table = [ uv_table_vertex_ref[vex_idx] for face in tqdm(temp_sub_mesh.triangles, desc="Building vu table") for vex_idx in face ]
		# * Normalize the uv_table
		dist_min = np.min([np.min(temp_sub_mesh.vertices, axis=0), np.min(temp_sub_pcd.points, axis=0)], axis=0)[:2]
		dist_max = np.max([np.max(temp_sub_mesh.vertices, axis=0), np.max(temp_sub_pcd.points, axis=0)], axis=0)[:2]
		# ! Ensure everything will be [0 ~ 1) after normalization
		dist_min -= self.config["voxel_size"]
		dist_max += self.config["voxel_size"]
		# If we have the "normalize_scale_factor" from caller
		dist_max_min = np.max(normalize_scale_factor) if normalize_scale_factor is not None else np.max(dist_max - dist_min)
		glog.debug(f"Using [{dist_max_min}] to normalize the uv table")
		uv_table = (uv_table - dist_min) / dist_max_min

		# Overlap map for detecting the occupancy?
		# Convert to rectangles
		cluster_ids, cluster_triangle_numbers, cluster_triangle_normals = temp_sub_mesh.cluster_connected_triangles()
		cluster_ids = np.array(cluster_ids)
		# Sort the cluster_ids by cluster_triangle_numbers
		sorted_cluster_ids = np.argsort(cluster_triangle_numbers)[::-1]

		cluster_info_dict = {} 
		# Processing each cluster
		for cluster_id in tqdm(sorted_cluster_ids, desc="Processing cluster meshes"):
			selects = cluster_ids == cluster_id
			selects = np.where(selects)[0]

			vertex_ids = [ vid for idx in selects for vid in temp_sub_mesh.triangles[idx]]
			cluster_mesh_voxel = o3d.geometry.VoxelGrid.create_from_triangle_mesh(temp_sub_mesh.select_by_index(vertex_ids, cleanup=False), self.config["voxel_size"])

			cluster_voxel_rect = rect_info(cluster_mesh_voxel.get_min_bound()[:2], (cluster_mesh_voxel.get_max_bound() - cluster_mesh_voxel.get_min_bound())[:2])
			cluster_voxel_rect.origin = (cluster_voxel_rect.origin - dist_min) / dist_max_min
			cluster_voxel_rect.length /= dist_max_min

			# ! Rotate 90 degree here
			rect_texture = np.zeros((*cluster_voxel_rect.get_length_in_pixel(texture_resolution, (1, 0)), 3), dtype=np.uint8)
			rect_texture[:] = self.config["default_color"]
			rect_mask = np.zeros(rect_texture.shape[:2], dtype=bool)

			pcd_selects = cluster_mesh_voxel.check_if_included(temp_sub_pcd.points)
			pcd_selects = np.where(pcd_selects)[0]
			cluster_pcd = temp_sub_pcd.select_by_index(pcd_selects)
			colors = cluster_pcd.colors
			uvs = (np.array(cluster_pcd.points)[:, :2] - dist_min) / dist_max_min
			uvs = (texture_resolution - 1) * (uvs - cluster_voxel_rect.origin)
			uvs = uvs.astype(np.int64)
			# ! X --> v, Y --> u
			rect_texture, rect_mask = set_texture_and_mask(uvs, colors, rect_texture, rect_mask)

			# TODO: Merge first before refinement
			rect_texture = np.asarray(self.refine_texture(rect_texture, rect_mask, self.config["texture_refinement_mask_size"])) if np.any(rect_mask) else rect_texture
			cluster_info_dict.update({cluster_id: cluster_info(cluster_id, selects, cluster_voxel_rect, rect_texture, rect_mask)})

		# * Rectangle packing / texture atlas
		glog.info("Running rectangle packing ...")
		scale_factor = 1.0
		attempts = 4
		for attmpt in range(attempts + 1):
			try:
				packed_positions = rpack.pack([ item.voxel_rect_info.get_length_in_pixel(texture_resolution, (1, 0)) for _, item in cluster_info_dict.items() ], max_width = texture_resolution, max_height = texture_resolution)
			except rpack.PackingImpossibleError:
				# * Handles the impossible packing cases
				scale_factor *= 2
				texture_resolution *= 2
				glog.info(f"Impossible packing, try to increase the texture resolution to {texture_resolution}")
				if attmpt >= attempts:
					raise rpack.PackingImpossibleError

		packed_size = np.max(np.array(packed_positions), axis=0) + 1
		glog.debug(f"{packed_size = }")
		glog.info("Ran rectangle packing")

		if scale_factor > 1.0:
			glog.info(f"{scale_factor = }")
			for _, info in cluster_info_dict.items():
				info.voxel_rect_info.apply_scale(1 / scale_factor)

		packed_texture = np.zeros((texture_resolution, texture_resolution, 3), dtype=np.uint8)
		packed_texture[:] = self.config["default_color"]
		packed_mask = np.zeros((texture_resolution, texture_resolution), dtype=bool)
		packed_uv_table = np.empty_like(uv_table)
		for packed_pos, (_, info) in tqdm(zip(packed_positions, cluster_info_dict.items()), desc="Packing texture"):
			_, triangle_selects, voxel_rect_info, rect_texture, rect_mask = info

			packed_texture[packed_pos[0] : packed_pos[0] + rect_texture.shape[0], packed_pos[1] : packed_pos[1] + rect_texture.shape[1] ] = rect_texture
			packed_mask[packed_pos[0] : packed_pos[0] + rect_mask.shape[0], packed_pos[1] : packed_pos[1] + rect_mask.shape[1] ] = rect_mask
			# packed_pos to uv
			packed_uv = (np.array(packed_pos)[[1, 0]] + 0.5)/ (texture_resolution - 1)
			offset = (packed_uv - voxel_rect_info.origin)
			uv_selects = [ select for triangle_select in triangle_selects for select in range(triangle_select * 3, (triangle_select + 1) * 3) ]
			packed_uv_table[uv_selects] = uv_table[uv_selects] + offset
		packed_texture = o3d.geometry.Image(packed_texture)

		return packed_uv_table, packed_texture, packed_mask

	def divide_by_plane(self, source_mesh: o3d.geometry.TriangleMesh, ref_colored_pcd: o3d.geometry.PointCloud, plane_vector: np.array, collect_everything: bool=False) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
		"""
		Divide the triangles into six categories by referring to the angles between triangle normals and target planes normals (/axes)

		Arguments:
			source_mesh: a mesh in "o3d.geometry.TriangleMesh" format
			ref_colored_pcd: a point cloud in "o3d.geometry.PointCloud" format
			plane_vector:
			collect_everything: if true mean it will collect all the triangles (useful if it's the last plane, in case we miss some triangles after 6 planes)
		"""
		glog.info("Computing the rotation vectors ... ")
		rot_vecs, thetas = rotation_between_vectors(np.tile(plane_vector, (len(source_mesh.triangle_normals), 1)), np.matrix(source_mesh.triangle_normals))
		glog.info("Computed the rotation vectors")

		sub_mesh = copy.deepcopy(source_mesh)

		# * Collect a sub mesh
		select_mask = thetas < np.deg2rad(45) if not collect_everything else np.array([True] * len(source_mesh.triangle_normals))
		select = np.where(select_mask)[0]
		sub_mesh.triangles = o3d.utility.Vector3iVector(np.array([sub_mesh.triangles[idx] for idx in select]).astype(np.int32))
		sub_mesh.triangle_normals = o3d.utility.Vector3dVector(np.array([sub_mesh.triangle_normals[idx] for idx in select]).astype(np.double))
		# TBC: Don't know why I cannot use the line below?
		# source_mesh.triangles = o3d.utility.Vector3iVector(source_mesh.triangles[select])

		# * Collect the remaining mesh
		remain_select = np.where(~select_mask)[0]
		remain_mesh = source_mesh
		if len(remain_select) > 0:
			remain_mesh.triangles = o3d.utility.Vector3iVector(np.array([source_mesh.triangles[idx] for idx in remain_select]).astype(np.int32))
			remain_mesh.triangle_normals = o3d.utility.Vector3dVector(np.array([source_mesh.triangle_normals[idx] for idx in remain_select]).astype(np.double))
		else:
			remain_mesh.triangles = o3d.utility.Vector3iVector()
			remain_mesh.triangle_normals = o3d.utility.Vector3dVector()

		glog.debug(f"Will processing {len(select)} faces, {len(remain_select)} will be left")

		sub_mesh_voxel = o3d.geometry.VoxelGrid.create_from_triangle_mesh(sub_mesh, self.config["voxel_size"])

		# * Collect sub point cloud
		glog.info("Collecting sub point cloud ...")
		selects = sub_mesh_voxel.check_if_included(ref_colored_pcd.points)
		sub_pcd = ref_colored_pcd.select_by_index(np.where(selects)[0])
		glog.info("Collected sub point cloud")

		return sub_mesh, sub_pcd, remain_mesh


	def process_six_projection_planes(self, target_planes: List[np.array]=None):
		"""
		**Perform the six projection planes method**

			(1) Divides triangles into 6 categories

			For each category:

				(2) Converts mesh into voxel, then we can filter out the outlier by using voxel

				(3) Transforms the sub mesh and sub point cloud w.r.t plane (/camera /projection ray)

				(4) Builds the uv table by referring to the point cloud x and y (w.r.t plane)

				(5) Projects the sub point cloud onto a plane, which is our final texture

				(6) Inpaints (/fills) the "holes" by fusion the colors with distance weight
		"""
		# * Rotate the target mesh
		pre_transform = None
		if self.config["align_the_axis"]:
			# TBC: Do we need to make the source_mesh immutable?
			obb = self.source_mesh.get_oriented_bounding_box()
			glog.info(f"{obb = }")
			glog.info(f"{obb.R = }")
			pre_transform = (obb.R.T, obb.center)
			self.source_mesh.rotate(*pre_transform)
			self.ref_colored_pcd.rotate(*pre_transform)

		# * Categorize the faces into 6 planes by referring the normal of faces
		if len(self.source_mesh.triangle_normals) == 0:
			self.source_mesh.compute_triangle_normals()

		glog.info(f"{self.source_mesh = }")
		glog.info(f"{self.ref_colored_pcd = }")

		target_planes = self.default_target_planes if target_planes is None else target_planes
		assert(len(target_planes) > 0)
		# ! Only for orthogonal projection planes approaches, like this six projection planes approach
		# ! Doesn't work for other non-orthogonal projection planes approaches
		np.max(self.ref_colored_pcd.get_max_bound() - self.ref_colored_pcd.get_min_bound())
		np.max(self.source_mesh.get_max_bound() - self.source_mesh.get_min_bound())
		# o3d.geometry.VoxelGrid.create_from_triangle_mesh(temp_sub_mesh.select_by_index(vertex_ids, cleanup=False), self.config["voxel_size"])
		normalize_scale_factor = np.max(self.source_mesh.get_max_bound() - self.source_mesh.get_min_bound())

		# * Reference information
		meters_per_pixel = normalize_scale_factor / self.config["texture_resolution"]
		glog.info(f"{meters_per_pixel = }")

		merged_mesh = o3d.geometry.TriangleMesh()
		# remain_mesh = self.source_mesh
		remain_mesh = copy.deepcopy(self.source_mesh)
		for idx, (plane_vector, alts_name) in enumerate(target_planes):
			glog.info(f"Processing plane [{alts_name}] ... {len(remain_mesh.triangle_normals)} faces left")

			# * Get sub mesh and sub point cloud for each projection plane
			if (len(target_planes) > 1):
				collect_all = alts_name == target_planes[-1][1] # Collect everything, if it's the last one
				sub_mesh, sub_pcd, remain_mesh = self.divide_by_plane(remain_mesh, self.ref_colored_pcd, plane_vector, collect_all)
			else:
				sub_mesh = remain_mesh
				sub_pcd = self.ref_colored_pcd

			if self.config["clustering_meshes"]:
				# ! Ensure everything will be [0 ~ 1) after normalization
				normalize_scale_factor += self.config["voxel_size"] * 2
				uv_table, texture, mask = self.project_mesh_cluster_to_plane(plane_vector, sub_mesh, sub_pcd, self.config["texture_resolution"], normalize_scale_factor)
			else:
				# * Project to the plane
				uv_table, texture, mask = self.project_to_plane(plane_vector, alts_name, sub_mesh, sub_pcd, self.config["texture_resolution"], normalize_scale_factor)

				texture = self.refine_texture(alts_name, texture, mask, self.config["texture_refinement_mask_size"])
			sub_mesh.vertex_colors = o3d.utility.Vector3dVector() # Remove vertex colors
			sub_mesh.triangle_normals = o3d.utility.Vector3dVector() # Remove face normals
			sub_mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_table) # Assign uv table to the sub mesh
			texture_np_rgb = cv2.cvtColor(np.asarray(texture), cv2.COLOR_BGR2RGB)
			sub_mesh.textures = [o3d.geometry.Image(texture_np_rgb)]
			# ! For the material ids of each sub mesh, just keep as 0
			sub_mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(sub_mesh.triangles))

			# * Merge these six planes into a full mesh
			merged_mesh = merged_mesh + sub_mesh if idx > 0 else sub_mesh

			if self.output_debug_files:
				sub_mesh_output_path = self.output_dir / f"sub_mesh_{alts_name}.obj"
				sub_pcd_output_path = self.output_dir / f"sub_pcd_{alts_name}.ply"
				glog.info(f"Writing debug mesh obj to {sub_mesh_output_path} ...")
				# ! Don't forget to rotate "back"
				if pre_transform is not None:
					sub_mesh.rotate(pre_transform[0].T, pre_transform[1])
					sub_pcd.rotate(pre_transform[0].T, pre_transform[1])
				o3d.io.write_triangle_mesh(str(sub_mesh_output_path), sub_mesh)
				o3d.io.write_point_cloud(str(sub_pcd_output_path), sub_pcd)
				glog.info("Wrote debug mesh obj")

			glog.info(f"Processed plane [{alts_name}]")

		# TBC: Why do we alway have some unprocessed faces left?
		glog.debug(f"Still have {len(remain_mesh.triangle_normals)} unprocessed faces")
		# * Write the merged sub meshes into a single mesh file
		glog.info(f"Writing final mesh obj to {self.output_dir} ...")
		o3d.io.write_triangle_mesh(str(self.output_dir / "textured_mesh.obj"), merged_mesh)
		glog.info("Wrote final mesh obj")

	def run(self):
		glog.info("Loading data ...")
		self.source_mesh = o3d.io.read_triangle_mesh(str(self.source_mesh_path))
		self.ref_colored_pcd = o3d.io.read_point_cloud(str(self.reference_pcd_path))
		glog.info("Loaded data")

		self.auto_parameterization()
		self.process_six_projection_planes([ self.default_target_planes[0] ] if self.config["single_projection_plane"] else None)
		