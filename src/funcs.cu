#include <stdio.h>
#include "funcs.h"

// FIXME: 无法使用
template<typename T>
__global__ void _addNumberInplaceKernel(T* data, size_t W, size_t H, double number)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	data[idx] += number;
};

// FIXME: 无法使用
template<typename T>
void _addNumberInplace(T** data, size_t fLen, size_t C, double val)
{
	T* imCuda;
	cudaMalloc((void**)&imCuda, fLen * C * sizeof(T));
	cudaMemcpy(imCuda, data, fLen * C * sizeof(T), cudaMemcpyHostToDevice);
	_addNumberInplaceKernel<T><<<fLen, C>>>(imCuda, fLen, C, val);
	cudaDeviceSynchronize();
	cudaMemcpy(data, imCuda, fLen * C * sizeof(T), cudaMemcpyDeviceToHost);
	cudaFree(imCuda);
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
