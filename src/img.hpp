#pragma once

#include <iostream>
#include <cstring>
#include <cmath>
#include <exception>
#include <limits>
#include <malloc.h>
#include "cpconfig.h"

#undef DLL_SPEC
#undef MUSIZE
#if defined (WIN32)
#define DLL_DEFINE
#define MUSIZE(x) _msize(x)
#if defined (DLL_DEFINE)
#define DLL_SPEC _declspec(dllexport)
#else
#define DLL_SPEC _declspec(dllimport)
#endif
#else
#define DLL_SPEC
#define MUSIZE(x) malloc_usable_size(x)
#endif

namespace tcv
{
	// 类型定义
	namespace type
	{
		typedef unsigned char U8;        // [0             , 255          )
		typedef unsigned int S8;         // [-128          , 127          )
		typedef unsigned short int U16;  // [0             , 65535        )
		typedef short int S16;           // [-32768        , 32767        )
		typedef int S32;                 // [-2147483648   , 2147483647   )
		typedef float F32;               // [1.18*10^{-38} , 3.40*10^{38} )
		typedef double F64;              // [2.23*10^{-308}, 1.79*10^{308})
	}

	// 图像类
	template <typename T, size_t C>
	class DLL_SPEC Img
	{
	private:
		size_t _height;       // 图像高度
		size_t _width;        // 图像宽度
		size_t _channels;     // 通道数
		T** _data;            // 存放数据
		mutable int _refNum;  // 引用计数

		// 打印数据
		template<typename U, size_t N>
		DLL_SPEC friend std::ostream& operator<<(
			std::ostream& os, const Img<U, N>& im);

	public:
		// 构造函数
		Img<T, C>(size_t height, size_t width)
			: _height(height), _width(width), _channels(C),
			_data(new T* [C]), _refNum(1)
		{
			for (int i = 0; i < _channels; ++i)
				_data[i] = new T[_height * _width];
		}
		Img<T, C>(size_t height, size_t width, T value)
			: _height(height), _width(width), _channels(C),
			_data(new T* [C]), _refNum(1)
		{
			size_t fLen = _height * _width;
			for (int i = 0; i < _channels; ++i)
			{
				_data[i] = new T[fLen];
				for (int j = 0; j < fLen; ++j)
					_data[i][j] = value;
			}
		}
		Img<T, C>(size_t height, size_t width, T** data)
		{
			size_t imCh = MUSIZE(data) / sizeof(*data);
			size_t imfLen = MUSIZE(*data) / sizeof(**data);
			size_t fLen = height * width;
			if (C > imCh)
				throw std::range_error("[Error] Invalid im's channels.");
			if (fLen > imfLen)
				throw std::range_error("[Error] Invalid im's size.");
			_height = height;
			_width = width;
			_channels = C;
			_refNum = 1;
			_data = new T * [_channels];
			for (int i = 0; i < _channels; ++i)
			{
				_data[i] = new T[fLen];
				for (int j = 0; j < fLen; ++j)
					_data[i][j] = data[i][j];
			}
		}
		// 复制构造函数，深拷贝
		Img<T, C>(const Img<T, C>& im)
		{
			if (im._channels != C)
				throw std::logic_error("[Error] Invalid im's channels.");
			if (strcmp(type(), im.type()) != 0)
				throw std::logic_error("[Error] Invalid im's type.");
			_height = im._height;
			_width = im._width;
			_channels = im._channels;
			_refNum = 1;
			size_t fLen = _height * _width;
			_data = new T * [_channels];
			for (int i = 0; i < _channels; ++i)
			{
				_data[i] = new T[fLen];
				for (int j = 0; j < fLen; ++j)
					_data[i][j] = im._data[i][j];
			}
		}
		// 析构函数
		~Img<T, C>()
		{
			if (--_refNum == 0)
			{
				for (int i = 0; i < _channels; ++i)
					delete[] _data[i];
				delete[] _data;
			}
		}
		// 赋值操作，浅拷贝
		Img<T, C>& operator=(const Img<T, C>& im)
		{
			if(this != &im)
			{
				if (im._channels != C)
					throw std::logic_error("[Error] Invalid im's channels.");
				if (strcmp(type(), im.type()) != 0)
					throw std::logic_error("[Error] Invalid im's type.");
				if (--_refNum == 0)
				{
					for (int i = 0; i < _channels; ++i)
						delete[] _data[i];
					delete[] _data;
				}
				_height = im._height;
				_width = im._width;
				_data = im._data;
				_refNum = im._refNum++;
			}
			return *this;
		}
		// +=操作符重载
		Img<T, C>& operator+=(const Img<T, C>& im)
		{
			if (!sameAs(im))
				throw std::logic_error("[Error] Mismatched img.");
			size_t fLen = im._height * im._width;
			for (int i = 0; i < im._channels; ++i)
				for (int j = 0; j < fLen; ++j)
					_data[i][j] = T(_data[i][j] + im._data[i][j]);
			return *this;
		}
		Img<T, C>& operator+=(T val)
		{ 
			size_t fLen = _height * _width;
			for (int i = 0; i < _channels; ++i)
				for (int j = 0; j < fLen; ++j)
					_data[i][j] = T(_data[i][j] + val);
			return *this;
		}
		// -=操作符重载
		Img<T, C>& operator-=(const Img<T, C>& im) 
		{ 
			if (!sameAs(im))
				throw std::logic_error("[Error] Mismatched img.");
			size_t fLen = im._height * im._width;
			for (int i = 0; i < im._channels; ++i)
				for (int j = 0; j < fLen; ++j)
					_data[i][j] = T(_data[i][j] - im._data[i][j]);
			return *this;
		}
		Img<T, C>& operator-=(T val) 
		{ 
			size_t fLen = _height * _width;
			for (int i = 0; i < _channels; ++i)
				for (int j = 0; j < fLen; ++j)
					_data[i][j] = T(_data[i][j] - val);
			return *this;
		}
		// *=操作符重载
		Img<T, C>& operator*=(const Img<T, C>& im) 
		{ 
			if (!sameAs(im))
				throw std::logic_error("[Error] Mismatched img.");
			size_t fLen = im._height * im._width;
			for (int i = 0; i < im._channels; ++i)
				for (int j = 0; j < fLen; ++j)
					_data[i][j] = T(_data[i][j] * im._data[i][j]);
			return *this;
		}
		Img<T, C>& operator*=(T val) 
		{ 
			size_t fLen = _height * _width;
			for (int i = 0; i < _channels; ++i)
				for (int j = 0; j < fLen; ++j)
					_data[i][j] = T(_data[i][j] * val);
			return *this;
		}
		// /=操作符重载
		Img<T, C>& operator/=(const Img<T, C>& im) 
		{ 
			if (!sameAs(im))
				throw std::logic_error("[Error] Mismatched img.");
			size_t fLen = im._height * im._width;
			for (int i = 0; i < im._channels; ++i)
				for (int j = 0; j < fLen; ++j)
					_data[i][j] = T(_data[i][j] / im._data[i][j]);
			return *this;
		}
		Img<T, C>& operator/=(T val) 
		{ 
			size_t fLen = _height * _width;
			for (int i = 0; i < _channels; ++i)
				for (int j = 0; j < fLen; ++j)
					_data[i][j] = T(_data[i][j] / val);
			return *this;
		}
		// 获取宽度
		size_t height() const { return _height; }
		// 获取高度
		size_t width() const { return _width; }
		// 获取通道数
		size_t channels() const { return _channels; }
		// 获取数据类型
		const char* type() const { return typeid(T).name(); }
		// 获取数据副本
		T** data(size_t ch = -1) const
		{
			size_t fLen = _height * _width;
			T** mats;
			if (ch == -1)
			{
				mats = new T * [_channels];
				for (int i = 0; i < _channels; ++i)
				{
					mats[i] = new T[fLen];
					for (int j = 0; j < fLen; ++j)
						mats[i][j] = _data[i][j];
				}
			}
			else
			{
				if (ch < 0 || ch >= _channels)
					throw std::out_of_range("[Error] Invalid `ch`.");
				mats = new T * [1];
				mats[0] = new T[fLen];
				for (int i = 0; i < fLen; ++i)
					mats[0][i] = _data[ch][i];
			}
			return mats;
		}
		// 获取某个像素值，不可修改
		T at(size_t col, size_t row, size_t ch = 0) const
		{
			if (ch < 0 || ch >= _channels)
				throw std::out_of_range("[Error] Invalid `ch`.");
			if (col < 0 || col >= _height)
				throw std::out_of_range("[Error] Invalid `col`.");
			if (row < 0 || row >= _width)
				throw std::out_of_range("[Error] Invalid `row`.");
			return _data[ch][col * _width + row];
		}
		// 获取某个像素值，可修改
		T& at(size_t col, size_t row, size_t ch = 0)
		{
			if (ch < 0 || ch >= _channels)
				throw std::out_of_range("[Error] Invalid `ch`.");
			if (col < 0 || col >= _height)
				throw std::out_of_range("[Error] Invalid `col`.");
			if (row < 0 || row >= _width)
				throw std::out_of_range("[Error] Invalid `row`.");
			return _data[ch][col * _width + row];
		}
		// 比较两个图像的信息是否相同，不包括具体数据
		bool sameAs(const Img<T, C>& im)
		{
			if (_height != im._height || _width != im._width || \
				_channels != im._channels || type() != im.type())
				return false;
			return true;
		}
		// 最大最小值计算
		void stateMinMax(double* min, double* max) 
		{
			*min = std::numeric_limits<T>::max();
			*max = std::numeric_limits<T>::min();
			size_t fLen = _height * _width;
			for (int i = 0; i < _channels; ++i)
				for (int j = 0; j < fLen; ++j)
				{
					if (_data[i][j] < *min)
						*min = _data[i][j];
					else if (_data[i][j] > *max)
						*max = _data[i][j];
				}
		}
		// 均值标准差计算
		void stateMeanStd(double* mean, double* stdev) 
		{
			*mean = 0;
			*stdev = 0;
			size_t fLen = _height * _width;
			double num = _channels * fLen;
			for (int i = 0; i < _channels; ++i)
				for (int j = 0; j < fLen; ++j)
					*mean += _data[i][j];
			*mean /= num;
			for (int i = 0; i < _channels; ++i)
				for (int j = 0; j < fLen; ++j)
					*stdev += std::pow(_data[i][j] - *mean, 2);
			*stdev = std::sqrt(*stdev / num);
		}
		// 新建等尺寸全0图像
		Img<T, C>& zerosLike()
		{
			return *(new Img<T, C>(_height, _width, 0));
		}
		// 新建等尺寸全1图像
		Img<T, C>& onesLike()
		{
			return *(new Img<T, C>(_height, _width, 1));
		}
	};

	template<typename U, size_t N>
	inline DLL_SPEC std::ostream& operator<<(std::ostream& os, const Img<U, N>& im)
	{
		os << "Shape: (" << im._height << ", " << im._width << ", "
			<< im._channels << ")\n";
		os << "Type: (" << im.type() << ")\n";
		os << "Array: [\n";
		for (int i = 0; i < im._channels; ++i)
		{
			os << "\tBand" << i + 1 << " [\n";
			for (int c = 0; c < im._height; ++c)
			{
				os << "\t\t";
				for (int r = 0; r < im._width; ++r)
				{
					if (strcmp(im.type(), typeid(type::U8).name()) == 0)
						os << int(im.at(c, r, i)) << " ";
					else
						os << im.at(c, r, i) << " ";
				}
				os << "\n";
			}
			os << "\t]\n";
		}
		os << "]";
		return os;
	};

	// 预定义图像
	typedef Img<type::U8, 1> ImgU8C1;
	typedef Img<type::U8, 2> ImgU8C2;
	typedef Img<type::U8, 3> ImgU8C3;
	typedef Img<type::U8, 4> ImgU8C4;
	typedef Img<type::S8, 1> ImgS8C1;
	typedef Img<type::S8, 2> ImgS8C2;
	typedef Img<type::S8, 3> ImgS8C3;
	typedef Img<type::S8, 4> ImgS8C4;
	typedef Img<type::U16, 1> ImgU16C1;
	typedef Img<type::U16, 2> ImgU16C2;
	typedef Img<type::U16, 3> ImgU16C3;
	typedef Img<type::U16, 4> ImgU16C4;
	typedef Img<type::S16, 1> ImgS16C1;
	typedef Img<type::S16, 2> ImgS16C2;
	typedef Img<type::S16, 3> ImgS16C3;
	typedef Img<type::S16, 4> ImgS16C4;
	typedef Img<type::S32, 1> ImgS32C1;
	typedef Img<type::S32, 2> ImgS32C2;
	typedef Img<type::S32, 3> ImgS32C3;
	typedef Img<type::S32, 4> ImgS32C4;
	typedef Img<type::F32, 1> ImgF32C1;
	typedef Img<type::F32, 2> ImgF32C2;
	typedef Img<type::F32, 3> ImgF32C3;
	typedef Img<type::F32, 4> ImgF32C4;
	typedef Img<type::F64, 1> ImgF64C1;
	typedef Img<type::F64, 2> ImgF64C2;
	typedef Img<type::F64, 3> ImgF64C3;
	typedef Img<type::F64, 4> ImgF64C4;
}
