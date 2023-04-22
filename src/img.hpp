#pragma once

#include <iostream>
#include <string>
#include <exception>
#include "cpconfig.h"

#undef DLL_SPEC
#undef MUSIZE
#if defined (WIN32)
#define MUSIZE(x) _msize(x)
#if defined (DLL_DEFINE)
#define DLL_SPEC _declspec(dllexport)
#else
#define DLL_SPEC _declspec(dllimport)
#endif
#else
#define DLL_SPEC
#include <malloc.h>
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
		DLL_SPEC friend std::ostream& operator<<(std::ostream& os, const Img<U, N>& im);
		////TODO: 比较图像
		//template<typename U, size_t N>
		//DLL_SPEC friend bool operator==(const Img& im1, const Img& im2);

	public:
		// 构造函数
		Img()
			: _height(0), _width(0), _channels(C), _data(new T* [C]), _refNum(1)
		{
			for (int i = 0; i < _channels; ++i)
				_data[i] = nullptr;
		}
		Img(size_t height, size_t width)
			: _height(height), _width(width), _channels(C),
			_data(new T* [C]), _refNum(1)
		{
			for (int i = 0; i < _channels; ++i)
				_data[i] = new T[_height * _width];
		}
		Img(size_t height, size_t width, T value)
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
		Img(size_t height, size_t width, T** data)
		{
			size_t imCh = MUSIZE(data) / sizeof(*data);
			size_t imfLen = MUSIZE(*data) / sizeof(**data);
			size_t fLen = height * width;
			if (C != imCh || fLen != imfLen)
				throw std::range_error("[Error] Invalid im's size or channels.");
			_height = height;
			_width = width;
			_channels = imCh;
			_data = new T * [_channels];
			_refNum = 1;
			for (int i = 0; i < _channels; ++i)
			{
				_data[i] = new T[fLen];
				for (int j = 0; j < fLen; ++j)
					_data[i][j] = data[i][j];
			}
		}
		Img(const Img& im)
		{
			if (im._channels != C)
				throw std::logic_error("[Error] Invalid im's channels.");
			_height = im._height;
			_width = im._width;
			_channels = im._channels;
			_data = new T * [_channels];
			_refNum = 1;
			size_t fLen = _height * _width;
			for (int i = 0; i < _channels; ++i)
			{
				_data[i] = new T[fLen];
				for (int j = 0; j < fLen; ++j)
					_data[i][j] = im._data[i][j];
			}
			im._refNum++;
		}
		// 析构函数
		~Img()
		{
			if (_data != nullptr && --_refNum == 0)
			{
				for (int i = 0; i < _channels; ++i)
					delete[] _data[i];
				delete[] _data;
			}
		}
		// 赋值操作
		Img& operator=(const Img& im)
		{
			if (this != &im)
			{
				if (_data != nullptr && --_refNum == 0)
				{
					for (int i = 0; i < _channels; ++i)
						delete[] _data[i];
					delete[] _data;
				}
				_channels = im._channels;
				return new Img(im);
			}
			return *this;
		}

		//// TODO: 操作符重载
		//void operator+(const Img& im) { }
		//void operator+(T val) { }
		//void operator-(const Img& im) { }
		//void operator-(T val) { }
		//void operator*(const Img& im) { }
		//void operator*(T val) { }
		//void operator/(const Img& im) { }
		//void operator/(T val) { }

		// 初始化
		void init(size_t height, size_t width)
		{
			_height = height;
			_width = width;
			for (int i = 0; i < _channels; ++i)
			{
				if (_data[i] != nullptr)
					delete[] _data[i];
				_data[i] = new T[_height * _width];
			}
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
		T at(size_t col, size_t row, size_t ch) const
		{
			if (col < 0 || col >= _height || row < 0 || row >= _width || \
				ch < 0 || ch >= _channels)
				throw std::out_of_range("[Error] Invalid `col` or `row` or `ch`.");
			return _data[ch][col * _width + row];
		}
		// 获取某个像素值，可修改
		T& at(size_t col, size_t row, size_t ch)
		{
			if (col < 0 || col >= _height || row < 0 || row >= _width || \
				ch < 0 || ch >= _channels)
				throw std::out_of_range("[Error] Invalid `col` or `row` or `ch`.");
			return _data[ch][col * _width + row];
		}

		//// TODO: 统计方法
		//void stateMinMax(double* min, double* max) { }
		//void stateMeanStd(double* mean, double* std) { }

		// 新建等尺寸全0图像
		static Img& zerosLike(const Img& im)
		{
			return new Img(im._height, im._width, 0);
		}
		// 新建等尺寸全1图像
		static Img& onesLike(const Img& im)
		{
			return new Img(im._height, im._width, 1);
		}
	};

	template<typename U, size_t N>
	DLL_SPEC inline std::ostream& operator<<(std::ostream& os, const Img<U, N>& im)
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
					if (strcmp(im.type(), "unsigned char") == 0)
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
	}

	// 预定义常规图像
	typedef Img<type::U8, 1> ImgGray;
	typedef Img<type::U8, 3> ImgRGB;
	typedef Img<type::U8, 4> ImgRGBA;
}
