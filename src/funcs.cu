#include <stdio.h>
#include "funcs.h"
#include "error.cuh"

// FIXME: 每次结果都不相同且结果错误，应该是内存复制错误
template <typename T>
__global__ void RGB2GRAYKernel(T** rgbData, T* grayData, size_t fLen)
{
	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < fLen)
	{
		grayData[idx] = static_cast<T>(
			0.299 * rgbData[0][idx] + \
			0.587 * rgbData[1][idx] + \
			0.114 * rgbData[2][idx]
			);
	}
};

template <typename T>
T** _RGB2GRAY(T** data, size_t fLen)
{
	// 分配主机内存
	T** grayCpu;
	grayCpu = new T * [1];
	grayCpu[0] = new T[fLen];
	// 分配设备内存并拷贝主机RGB数据到设备内存
	T** rgbCuda;
	T** hPtrs = new T * [3];
	for (int i = 0; i < 3; ++i)
	{
		T* tmpCuda;
		CHECK(cudaMalloc(&tmpCuda, sizeof(T) * fLen));
		CHECK(cudaMemcpy(tmpCuda, data[i], sizeof(T) * fLen, cudaMemcpyHostToDevice));
		hPtrs[i] = tmpCuda;
	}
	CHECK(cudaMalloc(&rgbCuda, sizeof(T*) * 3));
	CHECK(cudaMemcpy(rgbCuda, hPtrs, sizeof(T*) * 3, cudaMemcpyHostToDevice));
	// 为灰度图分配设备内存
	T* grayCuda;
	CHECK(cudaMalloc(&grayCuda, sizeof(T) * fLen));
	// 计算
	int blockSize = 128;
	int gridSize = (fLen - 1) / blockSize + 1;
	CHECK(cudaDeviceSynchronize());
	RGB2GRAYKernel<<<gridSize, blockSize>>>(rgbCuda, grayCuda, fLen);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	// 拷贝数据回主机
	CHECK(cudaMemcpy(grayCpu[0], grayCuda, sizeof(T) * fLen, cudaMemcpyDeviceToHost));
	// 清理
	CHECK(cudaFree(rgbCuda));
	CHECK(cudaFree(grayCuda));
	delete[] hPtrs;
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
