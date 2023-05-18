#include <stdio.h>
#include "funcs.h"
#include "error.cuh"

// FIXME: 每次结果都不相同且结果错误
template <typename T>
__global__ void RGB2GRAYKernel(T* data, T* grayData, size_t fLen, size_t pitch)
{
	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < fLen)
	{
		T r = *((data + 0 * pitch) + idx);
		T g = *((data + 1 * pitch) + idx);
		T b = *((data + 2 * pitch) + idx);
		printf("%d: %d, %d, %d\n", idx, r, g, b);
		grayData[idx] = static_cast<T>(0.2989 * r + 0.5870 * g + 0.1140 * b);
	}
}

template <typename T>
T** _RGB2GRAY(T** data, size_t fLen)
{
	T** grayCpu;
	grayCpu = new T * [1];
	grayCpu[0] = new T[fLen];
	T* rgbCuda, * grayCuda;
	size_t pitch;
	size_t size = sizeof(T) * fLen;
	CHECK(cudaMallocPitch(&rgbCuda, &pitch, size, 3));
	CHECK(cudaMemcpy2D(rgbCuda, pitch, data, size, size, 3, cudaMemcpyHostToDevice));
	CHECK(cudaMalloc(&grayCuda, size));
	int blockSize = 128;
	int gridSize = (fLen - 1) / blockSize + 1;
	CHECK(cudaDeviceSynchronize());
	RGB2GRAYKernel<<<gridSize, blockSize>>>(rgbCuda, grayCuda, fLen, pitch);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(grayCpu[0], grayCuda, size, cudaMemcpyDeviceToHost));
	CHECK(cudaFree(rgbCuda));
	CHECK(cudaFree(grayCuda));
	return grayCpu;
};

// 模板特例化
template tcv::type::U8** _RGB2GRAY<tcv::type::U8>(
	tcv::type::U8** data, size_t fLen);
template tcv::type::S8** _RGB2GRAY<tcv::type::S8>(
	tcv::type::S8** data, size_t fLen);
template tcv::type::U16** _RGB2GRAY<tcv::type::U16>(
	tcv::type::U16** data, size_t fLen);
template tcv::type::S16** _RGB2GRAY<tcv::type::S16>(
	tcv::type::S16** data, size_t fLen);
template tcv::type::S32** _RGB2GRAY<tcv::type::S32>(
	tcv::type::S32** data, size_t fLen);
template tcv::type::F32** _RGB2GRAY<tcv::type::F32>(
	tcv::type::F32** data, size_t fLen);
template tcv::type::F64** _RGB2GRAY<tcv::type::F64>(
	tcv::type::F64** data, size_t fLen);
