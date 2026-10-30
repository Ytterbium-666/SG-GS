/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <float.h>
#include <math_constants.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

__device__ __forceinline__ float3 pixelToWorldRay(
    float x,
    float y,
    float fx,
    float fy,
    float cx,
    float cy,
    const float* viewmatrix
)
{
    const float rx = (x - cx) / fx;
    const float ry = (y - cy) / fy;

    return make_float3(
        viewmatrix[0] * rx + viewmatrix[1] * ry + viewmatrix[2],
        viewmatrix[4] * rx + viewmatrix[5] * ry + viewmatrix[6],
        viewmatrix[8] * rx + viewmatrix[9] * ry + viewmatrix[10]
    );
}

__global__ void computeGaussianWaterMaskCUDA(
    const int P,
    const int W,
    const int H,
    const float2* __restrict__ points_xy_image,
    const float4* __restrict__ conic_opacity,
    const int* __restrict__ radii,
    const float* __restrict__ water_mask,
	const float* __restrict__ means3D,
    const float* __restrict__ viewmatrix,
    const float* __restrict__ cam_pos,
    const float focal_x,
    const float focal_y,
    const float principal_x,
    const float principal_y,
    float* __restrict__ gaussian_water_mask,
	float* __restrict__ gaussian_dg,
	float* __restrict__ gaussian_nonwater_ratio
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P)
        return;

    if (radii[idx] <= 0)
    {
        gaussian_water_mask[idx] = -1.0f;
		gaussian_nonwater_ratio[idx] = -1.0f;
        return;
    }

	const size_t offset = static_cast<size_t>(idx) * 3;
	gaussian_dg[offset] = CUDART_NAN_F;
	gaussian_dg[offset + 1] = CUDART_NAN_F;
	gaussian_dg[offset + 2] = CUDART_NAN_F;

    const float2 xy = points_xy_image[idx];
    const int r = radii[idx];

    int xmin = max(0, (int)floorf(xy.x - r));
    int xmax = min(W - 1, (int)ceilf(xy.x + r));
    int ymin = max(0, (int)floorf(xy.y - r));
    int ymax = min(H - 1, (int)ceilf(xy.y + r));

    if (xmin > xmax || ymin > ymax)
    {
        gaussian_water_mask[idx] = -1.0f;
        return;
    }

    float4 con_o = conic_opacity[idx];

    bool has_water = false;
    bool has_non_water = false;
    bool has_valid_pixel = false;

	float best_dist2 = FLT_MAX;
    int best_x = -1;
    int best_y = -1;
	int nonwater_count = 0;
	int valid_pixel_count = 0;

    for (int y = ymin; y <= ymax; ++y)
    {
        for (int x = xmin; x <= xmax; ++x)
        {
            float dx = xy.x - x;
            float dy = xy.y - y;

            float power = -0.5f * (
                con_o.x * dx * dx +
                con_o.z * dy * dy
            ) - con_o.y * dx * dy;

            if (power > 0.0f)
                continue;

            float alpha = con_o.w * expf(power);

            if (alpha < 1.0f / 255.0f)
                continue;

            has_valid_pixel = true;

			++valid_pixel_count;

            float m = water_mask[y * W + x];

            if (m > 0.5f)
                has_water = true;
            else
			{
                has_non_water = true;
				++nonwater_count;
				const float dist2 = dx * dx + dy * dy;
				if (dist2 < best_dist2)
				{
					best_dist2 = dist2;
                    best_x = x;
                    best_y = y;
				}
			}

        }
    }

    if (!has_valid_pixel)
    {
        gaussian_water_mask[idx] = -1.0f;
        return;
    }

	if (has_water && has_non_water)
	{
		gaussian_water_mask[idx] = 2.0f;
		gaussian_nonwater_ratio[idx] = static_cast<float>(nonwater_count) / static_cast<float>(valid_pixel_count);
	}

    if (has_water && !has_non_water)
    {
        gaussian_water_mask[idx] = 1.0f;
		return;
    }

    if (!has_water && has_non_water)
    {
        gaussian_water_mask[idx] = 0.0f;
		return;
    }

	if (best_x < 0 || best_y < 0)
	    return;

	if (!isfinite(focal_x) || !isfinite(focal_y) || focal_x <= 0.0f || focal_y <= 0.0f)
    {
        return;
    }	

	const float3 d_i = pixelToWorldRay(
        xy.x,
        xy.y,
        focal_x,
        focal_y,
        principal_x,
        principal_y,
        viewmatrix
    );

	const float3 d_nw = pixelToWorldRay(
        static_cast<float>(best_x),
        static_cast<float>(best_y),
        focal_x,
        focal_y,
        principal_x,
        principal_y,
        viewmatrix
    );

	const float norm_i = sqrtf(d_i.x * d_i.x + d_i.y * d_i.y + d_i.z * d_i.z);
	const float norm_nw = sqrtf(d_nw.x * d_nw.x + d_nw.y * d_nw.y + d_nw.z * d_nw.z);

	const float parallel_eps = 1e-6f;

	if (!isfinite(norm_i) || !isfinite(norm_nw) || norm_i <= 0.0f || norm_nw <= 0.0f || fabsf(d_i.z) <= parallel_eps * norm_i || fabsf(d_nw.z) <= parallel_eps * norm_nw)
    {
        return;
    }

	const float delta_z = means3D[offset + 2] - cam_pos[2];
	const float t_i = delta_z / d_i.z;
	const float t_nw = delta_z / d_nw.z;

	if (!isfinite(t_i) || !isfinite(t_nw))
        return;

	const float dg_x = t_nw * d_nw.x - t_i * d_i.x;
    const float dg_y = t_nw * d_nw.y - t_i * d_i.y;

	if (!isfinite(dg_x) || !isfinite(dg_y))
        return;

	gaussian_dg[offset] = dg_x;
    gaussian_dg[offset + 1] = dg_y;
	gaussian_dg[offset + 2] = 0.0f;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

__global__ void countMaxContribGaussianCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    const int W,
    const int H,
    const float2* __restrict__ means2D,
    const float4* __restrict__ conic_opacity,
    int* __restrict__ gaussian_count
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H)
        return;

    int tile_x = x / BLOCK_X;
    int tile_y = y / BLOCK_Y;
    int tile_id = tile_y * gridDim.x + tile_x;

    uint2 range = ranges[tile_id];

    float max_alpha = 0.0f;
    int max_gid = -1;

    for (int i = range.x; i < range.y; ++i)
    {
        int gid = point_list[i];

        float2 mean = means2D[gid];
        float2 d = { mean.x - x, mean.y - y };

        float4 con_o = conic_opacity[gid];

        float power = -0.5f * (
            con_o.x * d.x * d.x +
            con_o.z * d.y * d.y
        ) - con_o.y * d.x * d.y;

        if (power > 0.0f)
            continue;

        float alpha = con_o.w * expf(power);

        if (alpha > max_alpha)
        {
            max_alpha = alpha;
            max_gid = gid;
        }
    }

    if (max_gid >= 0)
        atomicAdd(&gaussian_count[max_gid], 1);
}


__global__ void generateBlurryMaskCUDA(
    const int P,
    const int* __restrict__ gaussian_count,
    const float threshold,
    float* __restrict__ blurry_mask
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= P)
        return;

    blurry_mask[idx] = gaussian_count[idx] > threshold ? 1.0f : 0.0f;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const float* medium_rgb,
	const float* medium_bs,
	const float* medium_attn,
	const float* water_mask,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* out_color_medium,
	float* out_medium,
	float* depth,
	float* gaussian_water_mask,
	float* gaussian_blurry_mask,
	float* gaussian_dg,
	float* gaussian_nonwater_count,
	bool antialiasing,
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		antialiasing
	), debug)

	if (water_mask != nullptr)
    {
		const float principal_x = 0.5f * (width - 1);
        const float principal_y = 0.5f * (height - 1);
		computeGaussianWaterMaskCUDA<<<(P + 255) / 256, 256>>>(
            P,
            width,
            height,
            geomState.means2D,
			geomState.conic_opacity,
			radii,
            water_mask,
			means3D,
            viewmatrix,
            cam_pos,
            focal_x,
            focal_y,
            principal_x,
            principal_y,
            gaussian_water_mask,
			gaussian_dg,
			gaussian_nonwater_count
        );
	    CHECK_CUDA(, debug)
	}

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;   // 读取总共需要渲染的 Gaussian instance 数量
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);  
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	// 对每个 Gaussian 覆盖的每个 tile，生成一个：key = [tile ID | depth] value = Gaussian ID
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)
	
	// 每个 ranges[tile] 是 uint2，保存 start/end
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	int* gaussian_count = nullptr;
    CHECK_CUDA(cudaMalloc(&gaussian_count, P * sizeof(int)), debug);
    CHECK_CUDA(cudaMemset(gaussian_count, 0, P * sizeof(int)), debug);

    dim3 pixel_block(BLOCK_X, BLOCK_Y);
    dim3 pixel_grid(
    (width + BLOCK_X - 1) / BLOCK_X,
    (height + BLOCK_Y - 1) / BLOCK_Y
    );

    countMaxContribGaussianCUDA<<<pixel_grid, pixel_block>>>(
    imgState.ranges,
    binningState.point_list,
    width,
    height,
    geomState.means2D,
    (float4*)geomState.conic_opacity,
    gaussian_count
    );
    CHECK_CUDA(, debug);

    float theta = 0.0002f;
    float threshold = theta * width * height;

    generateBlurryMaskCUDA<<<(P + 255) / 256, 256>>>(
    P,
    gaussian_count,
    threshold,
    gaussian_blurry_mask
    );
    CHECK_CUDA(, debug);

    CHECK_CUDA(cudaFree(gaussian_count), debug);

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		medium_rgb,
	    medium_bs,
	    medium_attn,
		water_mask,
		out_color,
		out_color_medium,
	    out_medium,
		geomState.depths,
		depth), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const float* medium_rgb,
	const float* medium_bs,
	const float* medium_attn,
	const float* water_mask,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_dpix_medium,
	const float* dL_dmedium,
	const float* dL_invdepths,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dinvdepth,
	float* dL_dmedium_rgb,
	float* dL_dmedium_bs,
	float* dL_dmedium_attn,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool antialiasing,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		geomState.depths,
		medium_rgb,
	    medium_bs,
	    medium_attn,
		water_mask,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_dpix_medium,
		dL_dmedium,
		dL_invdepths,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dinvdepth,
		dL_dmedium_rgb,
	    dL_dmedium_bs,
	    dL_dmedium_attn), debug);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		opacities,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		dL_dinvdepth,
		dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		antialiasing), debug);
}
