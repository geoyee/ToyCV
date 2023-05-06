#include <stdio.h>
#include "funcs.h"
#include "error.cuh"

template<typename T>
__global__ void _addNumberKernel(T** data, size_t fLen, size_t C, double val)
{
	size_t idx_x = threadIdx.x + blockIdx.x * blockDim.x;
	size_t idx_y = threadIdx.y + blockIdx.y * blockDim.y;
	if (idx_x < fLen && idx_y < C)
	{
		data[idx_y][idx_x] = static_cast<T>(data[idx_y][idx_x] + val);
	}
};

// FIXME: 36行 [700] 遇到非法内存访问
template<typename T>
void _addNumberInplace(T** data, size_t fLen, size_t C, double val)
{
	T** imCpu;
	imCpu = new T * [C];
	for (size_t i = 0; i < C; ++i)
		imCpu[i] = new T[fLen];
	T** imCuda;
	size_t pitch;
	size_t size = sizeof(T) * fLen;
	CHECK(cudaMallocPitch((void**)&imCuda, &pitch, size, C));
	CHECK(cudaMemset2D(imCuda, pitch, 0, size, C));
	CHECK(cudaMemcpy2D(imCuda, pitch, data, size, size, C, cudaMemcpyHostToDevice));
	dim3 block_size(16, 16);
	dim3 grid_size((fLen + block_size.x - 1) / block_size.x, (C + block_size.y - 1) / block_size.y);
	CHECK(cudaDeviceSynchronize());
	_addNumberKernel<T><<<grid_size, block_size>>>(imCuda, fLen, C, val);
	// 检查核函数需要在调用核函数后手动加上这两句
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy2D(imCpu, size, imCuda, pitch, size, C, cudaMemcpyDeviceToHost));
	CHECK(cudaFree(imCuda));
	for (int i = 0; i < 10; ++i)
		printf("old val: %d and new val: %d\n", data[0][i], imCpu[0][i]);
	data = imCpu;
};

// 模板特例化
template void _addNumberInplace<tcv::type::U8>(
	tcv::type::U8** data, size_t fLen, size_t C, double val);
template void _addNumberInplace<tcv::type::S8>(
	tcv::type::S8** data, size_t fLen, size_t C, double val);
template void _addNumberInplace<tcv::type::U16>(
	tcv::type::U16** data, size_t fLen, size_t C, double val);
template void _addNumberInplace<tcv::type::S16>(
	tcv::type::S16** data, size_t fLen, size_t C, double val);
template void _addNumberInplace<tcv::type::S32>(
	tcv::type::S32** data, size_t fLen, size_t C, double val);
template void _addNumberInplace<tcv::type::F32>(
	tcv::type::F32** data, size_t fLen, size_t C, double val);
template void _addNumberInplace<tcv::type::F64>(
	tcv::type::F64** data, size_t fLen, size_t C, double val);
