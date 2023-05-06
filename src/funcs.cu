#include <stdio.h>
#include "funcs.h"

// FIXME: 加法错误
template<typename T>
__global__ void _addNumberKernel(T** data, size_t W, size_t H, double number)
{
	const int y = threadIdx.x;
	const int x = blockIdx.x;
	data[y][x] = static_cast<T>(data[y][x] + number);
};

// FIXME: 无效205
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
	cudaMallocPitch((void**)&imCuda, &pitch, size, C);
	cudaMemset2D(imCuda, pitch, 0, size, C);
	cudaMemcpy2D(imCuda, pitch, data, size, size, C, cudaMemcpyHostToDevice);
	_addNumberKernel<T><<<fLen, C>>>(imCuda, fLen, C, val);
	cudaMemcpy2D(imCpu, size, imCuda, pitch, size, C, cudaMemcpyDeviceToHost);
	cudaFree(imCuda);
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
