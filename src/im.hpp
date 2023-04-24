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
	// 常用图像类型
	namespace type
	{
		typedef unsigned char U8;        // [0             , 255          )
		typedef signed char S8;          // [-128          , 127          )
		typedef unsigned short int U16;  // [0             , 65535        )
		typedef signed short int S16;    // [-32768        , 32767        )
		typedef signed int S32;          // [-2147483648   , 2147483647   )
		typedef float F32;               // [1.18*10^{-38} , 3.40*10^{38} )
		typedef double F64;              // [2.23*10^{-308}, 1.79*10^{308})
	}

	// 形状
	struct DLL_SPEC Shape
	{
		size_t height;    // 图像高度
		size_t width;     // 图像宽度
		size_t channels;  // 通道数

		// 打印形状
		DLL_SPEC friend std::ostream& operator<<(std::ostream& os, const Shape& shp)
		{
			os << "Shape: (" << shp.height << ", " << \
				shp.width << ", " << shp.channels << ")";
			return os;
		}
		// 比较两个形状是否相等
		DLL_SPEC friend bool operator==(const Shape& shp1, const Shape& shp2)
		{
			if (shp1.height == shp2.height && \
				shp1.width == shp2.width && \
				shp1.channels == shp2.channels)
				return true;
			else
				return false;
		}

		// 构造函数
		Shape()
			: height(0), width(0), channels(0) { }
		Shape(size_t h, size_t w, size_t c)
			: height(h), width(w), channels(c) { }
		// 复制构造函数
		Shape(const Shape& shp)
			: height(shp.height), width(shp.width), channels(shp.channels) { }
		// 析构函数
		~Shape() { }
		// 赋值操作
		Shape& operator=(const Shape& shp)
		{
			if (this != &shp)
			{
				height = shp.height;
				width = shp.width;
				channels = shp.channels;
			}
			return *this;
		}
	};

	// 图像
	template <typename T>
	class DLL_SPEC Im
	{
	private:
		Shape _shape;            // 图像形状
		T** _data;               // 存放数据
		mutable size_t _refNum;  // 引用计数

		// 打印信息
		template<typename U>
		DLL_SPEC friend std::ostream& operator<<(std::ostream& os, const Im<U>& im)
		{
			{
				os << im._shape << "\n";
				os << "Type: (" << im.type() << ")\n";
				os << "Array: [\n";
				for (size_t i = 0; i < im.channels(); ++i)
				{
					os << "\tBand" << i + 1 << " [\n";
					for (size_t c = 0; c < im.height(); ++c)
					{
						os << "\t\t";
						for (size_t r = 0; r < im.width(); ++r)
						{
							if (strcmp(im.type(), typeid(type::U8).name()) == 0 || \
								strcmp(im.type(), typeid(type::S8).name()) == 0)
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
		}
		// 加法重载
		template<typename U>
		DLL_SPEC friend Im<U>& operator+(const Im<U>& im1, const Im<U>& im2)
		{
			if (!im1.sameAs(im2))
				throw std::logic_error("[Error] Im1 not matched Im2.");
			Im<U>* newIm = new Im(im1);
			*newIm += im2;
			return *newIm;
		}
		// 减法重载
		template<typename U>
		DLL_SPEC friend Im<U>& operator-(const Im<U>& im1, const Im<U>& im2)
		{
			if (!im1.sameAs(im2))
				throw std::logic_error("[Error] Im1 not matched Im2.");
			Im<U>* newIm = new Im(im1);
			*newIm -= im2;
			return *newIm;
		}
		// 乘法重载
		template<typename U>
		DLL_SPEC friend Im<U>& operator*(const Im<U>& im1, const Im<U>& im2)
		{
			if (!im1.sameAs(im2))
				throw std::logic_error("[Error] Im1 not matched Im2.");
			Im<U>* newIm = new Im(im1);
			*newIm *= im2;
			return *newIm;
		}
		// 除法重载
		template<typename U>
		DLL_SPEC friend Im<U>& operator/(const Im<U>& im1, const Im<U>& im2)
		{
			if (!im1.sameAs(im2))
				throw std::logic_error("[Error] Im1 not matched Im2.");
			Im<U>* newIm = new Im(im1);
			*newIm /= im2;
			return *newIm;
		}

	public:
		// 构造函数
		Im(size_t h, size_t w, size_t c)
			: _shape(Shape(h, w, c)), _data(new T* [c]), _refNum(1)
		{
			for (size_t i = 0; i < c; ++i)
				_data[i] = new T[h * w];
		}
		Im(const Shape& shp)
			: _shape(Shape(shp)), _data(new T* [shp.channels]), _refNum(1)
		{
			for (size_t i = 0; i < shp.channels; ++i)
				_data[i] = new T[shp.height * shp.width];
		}
		Im(size_t h, size_t w, size_t c, T value)
			: _shape(Shape(h, w, c)), _data(new T* [c]), _refNum(1)
		{
			size_t fLen = h * w;
			for (size_t i = 0; i < c; ++i)
			{
				_data[i] = new T[fLen];
				for (size_t j = 0; j < fLen; ++j)
					_data[i][j] = value;
			}
		}
		Im(const Shape& shp, T value)
			: _shape(Shape(shp)), _data(new T* [shp.channels]), _refNum(1)
		{
			size_t fLen = shp.height * shp.width;
			for (size_t i = 0; i < shp.channels; ++i)
			{
				_data[i] = new T[fLen];
				for (size_t j = 0; j < fLen; ++j)
					_data[i][j] = value;
			}
		}
		Im(size_t h, size_t w, size_t c, T** data, bool chFirst = true)
		{
			size_t imCh, imfLen;
			if (chFirst)
			{
				imCh = MUSIZE(data) / sizeof(*data);
				imfLen = MUSIZE(*data) / sizeof(**data);
			}
			else
			{
				imfLen = MUSIZE(data) / sizeof(*data);
				imCh = MUSIZE(*data) / sizeof(**data);
			}
			size_t fLen = h * w;
			// FIXME: Linux下无法确保相等，先使用大于
			if (c > imCh)
				throw std::range_error("[Error] Invalid im's channels.");
			if (fLen > imfLen)
				throw std::range_error("[Error] Invalid im's size.");
			_shape = Shape(h, w, c);
			_refNum = 1;
			_data = new T * [c];
			for (size_t i = 0; i < c; ++i)
			{
				_data[i] = new T[fLen];
				for (size_t j = 0; j < fLen; ++j)
				{
					if (chFirst)
						_data[i][j] = data[i][j];
					else
						_data[i][j] = data[j][i];
				}
			}
		}
		Im(const Shape& shp, T** data, bool chFirst = true)
		{
			size_t imCh, imfLen;
			if (chFirst)
			{
				imCh = MUSIZE(data) / sizeof(*data);
				imfLen = MUSIZE(*data) / sizeof(**data);
			}
			else
			{
				imfLen = MUSIZE(data) / sizeof(*data);
				imCh = MUSIZE(*data) / sizeof(**data);
			}
			size_t fLen = shp.height * shp.width;
			// FIXME: Linux下无法确保相等，先使用大于
			if (shp.channels > imCh)
				throw std::range_error("[Error] Invalid im's channels.");
			if (fLen > imfLen)
				throw std::range_error("[Error] Invalid im's size.");
			_shape = Shape(shp);
			_refNum = 1;
			_data = new T * [shp.channels];
			for (size_t i = 0; i < shp.channels; ++i)
			{
				_data[i] = new T[fLen];
				for (size_t j = 0; j < fLen; ++j)
				{
					if (chFirst)
						_data[i][j] = data[i][j];
					else
						_data[i][j] = data[j][i];
				}
			}
		}
		// 复制构造函数，深拷贝
		Im(const Im<T>& im)
		{
			if (strcmp(type(), im.type()) != 0)
				throw std::logic_error("[Error] Invalid im's type.");
			_shape = Shape(im._shape);
			_refNum = 1;
			size_t fLen = im.height() * im.width();
			_data = new T * [im.channels()];
			for (size_t i = 0; i < im.channels(); ++i)
			{
				_data[i] = new T[fLen];
				for (size_t j = 0; j < fLen; ++j)
					_data[i][j] = im._data[i][j];
			}
		}
		// 析构函数
		~Im()
		{
			if (--_refNum == 0)
			{
				for (size_t i = 0; i < channels(); ++i)
					delete[] _data[i];
				delete[] _data;
			}
		}
		// 赋值操作，浅拷贝
		Im<T>& operator=(const Im<T>& im)
		{
			if (this != &im)
			{
				if (strcmp(type(), im.type()) != 0)
					throw std::logic_error("[Error] Invalid im's type.");
				if (--_refNum == 0)
				{
					for (size_t i = 0; i < channels(); ++i)
						delete[] _data[i];
					delete[] _data;
				}
				_shape = Shape(im._shape);
				_data = im._data;
				_refNum = im._refNum++;
			}
			return *this;
		}
		// 原址加法操作符重载
		Im<T>& operator+=(const Im<T>& im)
		{
			if (!sameAs(im))
				throw std::logic_error("[Error] Mismatched Im.");
			size_t fLen = im.height() * im.width();
			for (size_t i = 0; i < im.channels(); ++i)
				for (size_t j = 0; j < fLen; ++j)
					_data[i][j] = T(_data[i][j] + im._data[i][j]);
			return *this;
		}
		Im<T>& operator+=(T val)
		{
			size_t fLen = height() * width();
			for (size_t i = 0; i < channels(); ++i)
				for (size_t j = 0; j < fLen; ++j)
					_data[i][j] = T(_data[i][j] + val);
			return *this;
		}
		// 原址减法操作符重载
		Im<T>& operator-=(const Im<T>& im)
		{
			if (!sameAs(im))
				throw std::logic_error("[Error] Mismatched Im.");
			size_t fLen = im.height() * im.width();
			for (size_t i = 0; i < im.channels(); ++i)
				for (size_t j = 0; j < fLen; ++j)
					_data[i][j] = T(_data[i][j] - im._data[i][j]);
			return *this;
		}
		Im<T>& operator-=(T val)
		{
			size_t fLen = height() * width();
			for (size_t i = 0; i < channels(); ++i)
				for (size_t j = 0; j < fLen; ++j)
					_data[i][j] = T(_data[i][j] - val);
			return *this;
		}
		// 原址乘法操作符重载
		Im<T>& operator*=(const Im<T>& im)
		{
			if (!sameAs(im))
				throw std::logic_error("[Error] Mismatched Im.");
			size_t fLen = im.height() * im.width();
			for (size_t i = 0; i < im.channels(); ++i)
				for (size_t j = 0; j < fLen; ++j)
					_data[i][j] = T(_data[i][j] * im._data[i][j]);
			return *this;
		}
		Im<T>& operator*=(T val)
		{
			size_t fLen = height() * width();
			for (size_t i = 0; i < channels(); ++i)
				for (size_t j = 0; j < fLen; ++j)
					_data[i][j] = T(_data[i][j] * val);
			return *this;
		}
		// 原址除法操作符重载
		Im<T>& operator/=(const Im<T>& im)
		{
			if (!sameAs(im))
				throw std::logic_error("[Error] Mismatched Im.");
			size_t fLen = im.height() * im.width();
			for (size_t i = 0; i < im.channels(); ++i)
				for (size_t j = 0; j < fLen; ++j)
					_data[i][j] = T(_data[i][j] / im._data[i][j]);
			return *this;
		}
		Im<T>& operator/=(T val)
		{
			size_t fLen = height() * width();
			for (size_t i = 0; i < channels(); ++i)
				for (size_t j = 0; j < fLen; ++j)
					_data[i][j] = T(_data[i][j] / val);
			return *this;
		}
		// 获取形状
		const Shape& shape() const { return _shape; }
		// 获取宽度
		size_t height() const { return _shape.height; }
		// 获取高度
		size_t width() const { return _shape.width; }
		// 获取通道数
		size_t channels() const { return _shape.channels; }
		// 获取数据类型
		const char* type() const { return typeid(T).name(); }
		// 获取数据副本
		T** data() const
		{
			size_t fLen = height() * width();
			T** mats = new T * [channels()];
			for (size_t i = 0; i < channels(); ++i)
			{
				mats[i] = new T[fLen];
				for (size_t j = 0; j < fLen; ++j)
					mats[i][j] = _data[i][j];
			}
			return mats;
		}
		T** data(size_t ch) const
		{
			if (ch < 0 || ch >= channels())
				throw std::range_error("[Error] Invalid ch.");
			size_t fLen = height() * width();
			T** mats = new T * [1];
			mats[0] = new T[fLen];
			for (size_t i = 0; i < fLen; ++i)
				mats[0][i] = _data[0][i];
			return mats;
		}
		T** data(size_t x, size_t y, size_t xOff, size_t yOff) const
		{
			if (x < 0 || x >= height() || xOff < 0 || x + xOff >= height())
				throw std::range_error("[Error] Invalid x or xOff.");
			if (y < 0 || y >= width() || yOff < 0 || y + yOff >= width())
				throw std::range_error("[Error] Invalid y or yOff.");
			T** mats = new T * [channels()];
			for (int c = 0; c < channels(); ++c)
			{
				mats[c] = new T[xOff * yOff];
				size_t k = 0;
				for (size_t i = x; i < x + xOff; ++i)
					for (size_t j = y; j < y + yOff; ++j)
					{
						mats[0][k] = _data[0][i * width() + j];
						k++;
					}
			}
			return mats;
		}
		T** data(size_t x, size_t y, size_t xOff, size_t yOff, size_t ch) const
		{
			if (x < 0 || x >= height() || xOff < 0 || x + xOff >= height())
				throw std::range_error("[Error] Invalid x or xOff.");
			if (y < 0 || y >= width() || yOff < 0 || y + yOff >= width())
				throw std::range_error("[Error] Invalid y or yOff.");
			if (ch < 0 || ch >= channels())
				throw std::range_error("[Error] Invalid ch.");
			T** mats = new T * [1];
			mats[0] = new T[xOff * yOff];
			size_t k = 0;
			for (size_t i = x; i < x + xOff; ++i)
				for (size_t j = y; j < y + yOff; ++j)
				{
					mats[0][k] = _data[0][i * width() + j];
					k++;
				}
			return mats;
		}
		// 获取某个像素值，不可修改
		T at(size_t col, size_t row, size_t ch = 0) const
		{
			if (ch < 0 || ch >= channels())
				throw std::out_of_range("[Error] Invalid ch.");
			if (col < 0 || col >= height())
				throw std::out_of_range("[Error] Invalid col.");
			if (row < 0 || row >= width())
				throw std::out_of_range("[Error] Invalid row.");
			return _data[ch][col * width() + row];
		}
		// 获取某个像素值，可修改
		T& at(size_t col, size_t row, size_t ch = 0)
		{
			if (ch < 0 || ch >= channels())
				throw std::out_of_range("[Error] Invalid ch.");
			if (col < 0 || col >= height())
				throw std::out_of_range("[Error] Invalid col.");
			if (row < 0 || row >= width())
				throw std::out_of_range("[Error] Invalid row.");
			return _data[ch][col * width() + row];
		}
		// 比较两个图像的形状和类型是否相同
		bool sameAs(const Im<T>& im) const
		{
			if (_shape == im.shape() && type() == im.type())
				return true;
			return false;
		}
		// 最大最小值计算
		void stateMinMax(double* min, double* max) const
		{
			*min = std::numeric_limits<T>::max();
			*max = std::numeric_limits<T>::min();
			size_t fLen = height() * width();
			for (size_t i = 0; i < channels(); ++i)
				for (size_t j = 0; j < fLen; ++j)
				{
					if (_data[i][j] < *min)
						*min = _data[i][j];
					else if (_data[i][j] > *max)
						*max = _data[i][j];
				}
		}
		// 均值标准差计算
		void stateMeanStd(double* mean, double* stdev) const
		{
			*mean = 0;
			*stdev = 0;
			size_t fLen = height() * width();
			double num = channels() * double(fLen);
			for (size_t i = 0; i < channels(); ++i)
				for (size_t j = 0; j < fLen; ++j)
					*mean += _data[i][j];
			*mean /= num;
			for (size_t i = 0; i < channels(); ++i)
				for (size_t j = 0; j < fLen; ++j)
					*stdev += std::pow(_data[i][j] - *mean, 2);
			*stdev = std::sqrt(*stdev / num);
		}
		// 新建等尺寸全0图像
		Im<T>& zerosLike() const
		{
			return *(new Im<T>(_shape, 0));
		}
		// 新建等尺寸全1图像
		Im<T>& onesLike() const
		{
			return *(new Im<T>(_shape, 1));
		}
	};

	// 预定义图像
	typedef Im<type::U8> ImU8;
	typedef Im<type::S8> ImS8;
	typedef Im<type::U16> ImU16;
	typedef Im<type::S16> ImS16;
	typedef Im<type::S32> ImS32;
	typedef Im<type::F32> ImF32;
	typedef Im<type::F64> ImF64;
}
