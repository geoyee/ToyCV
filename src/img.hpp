#pragma once

#include <iostream>
#include <exception>

#undef DLL_SPEC
#if defined (WIN32)
#if defined (DLL_DEFINE)
#define DLL_SPEC _declspec(dllexport)
#else
#define DLL_SPEC _declspec(dllimport)
#endif
#else
#define DLL_SPEC
#endif

namespace tcv
{
	struct DLL_SPEC Shape
	{
		size_t height;  // 高度
		size_t width;   // 宽度
		int channel;    // 通道数

		friend std::ostream& operator<<(std::ostream& os, const Shape& shp)
		{
			os << "Shape (" << shp.height << ", "
				<< shp.width << ", "
				<< shp.channel << ")";
			return os;
		}

		Shape()
			: height(0), width(0), channel(0) { }
		Shape(size_t h, size_t w, int c)
			: height(h), width(w), channel(c) { }
		Shape(const Shape& shp)
			: height(shp.height), width(shp.width), channel(shp.channel) { }
		~Shape() { }
	};

	template <typename T, int C>
	class DLL_SPEC Img
	{
	private:
		tcv::Shape _shp;  // 图像形状
		T** _data;		  // 存放数据

		template<typename TT, int CC>
		friend std::ostream& operator<<(std::ostream& os, const Img<TT, CC>& img)
		{
			os << "Shape (" << img._shp.height << ", "
				<< img._shp.width << ", "
				<< img._shp.channel << ")" << "\n";
			os << "Array [\n";
			for (int i = 0; i < CC; ++i)
			{
				os << "[\n";
				for (int c = 0; c < img._shp.height; ++c)
				{
					for (int r = 0; r < img._shp.width; ++r)
						os << float(img.pixAt(c, r, i)) << " ";
					os << "\n";
				}
				os << "]\n";
			}
			os << "]";
			return os;
		}

	public:
		Img()
			: _shp(Shape(0, 0, C)), _data(nullptr) { }
		Img(size_t height, size_t width)
			: _shp(Shape(height, width, C)), _data(new T* [C])
		{
			for (int i = 0; i < C; ++i)
				_data[i] = new T[height * width];
		}
		Img(const tcv::Shape& shp)
			: _shp(Shape(shp.height, shp.width, C))
		{
			if (shp.channel != C)
				std::cerr << "[Warning] `shp` channel not equal `C`, \
							  `C` has been forcibly used." << std::endl;
			_data = new T * [C];
			for (int i = 0; i < C; ++i)
				_data[i] = new T[height * width];
		}
		Img(size_t height, size_t width, T value)
			: _shp(Shape(height, width, C)), _data(new T* [C])
		{
			for (int i = 0; i < C; ++i)
			{
				_data[i] = new T[height * width];
				for (int j = 0; j < height * width; ++j)
					_data[i][j] = value;
			}
		}
		Img(const tcv::Shape& shp, T value)
			: _shp(Shape(shp.height, shp.width, C))
		{
			_data = new T * [C];
			if (shp.channel != C)
				std::cerr << "[Warning] `shp` channel not equal `C`, \
							  `C` has been forcibly used." << std::endl;
			for (int i = 0; i < C; ++i)
			{
				_data[i] = new T[height * width];
				for (int j = 0; j < height * width; ++j)
					_data[i][j] = value;
			}
		}
		Img(const Img& im)
		{
			if (im._shp.channel != C)
				throw std::domain_error("[Error] `im` channels not equal `C`.");
			_shp = im._shp;
			_data = new T * [C];
			for (int i = 0; i < C; ++i)
			{
				_data[i] = new T[_shp.height * _shp.width];
				for (int j = 0; j < _shp.height * _shp.width; ++j)
					_data[i][j] = im._data[i][j];
			}
		}
		~Img()
		{
			if (_data != nullptr)
			{
				for (int i = 0; i < C; ++i)
					delete[] _data[i];
				delete[] _data;
			}
		}
		// 获取形状
		tcv::Shape shape() { return _shp; }
		size_t height() { return _shp.height; }
		size_t width() { return _shp.width; }
		size_t channel() { return _shp.channel; }
		// 获取数据
		T** data() { return _data; }
		T* matAt(int ci)
		{
			if (ci < 0 || ci >= C)
				throw std::out_of_range(
					"[Error] `ci` is less than 0 or greater than `C`.");
			return _data[ci];
		}
		T* chsAt(size_t col, size_t row)
		{
			if (col < 0 || col >= _shp.height || row < 0 || row >= _shp.width)
				throw std::out_of_range(
					"[Error] `col` or `row` out of range.");
			T* chs = new T[C];
			for (int i = 0; i < C; ++i)
				chs[i] = _data[i][col * _shp.width + row];
			return chs;
		}
		T pixAt(size_t col, size_t row, int ci) const
		{
			if (col < 0 || col >= _shp.height || row < 0 || row >= _shp.width || \
				ci < 0 || ci >= C)
				throw std::out_of_range(
					"[Error] `col` or `row` or `ci` out of range.");
			return _data[ci][col * _shp.width + row];
		}
		T& pixAt(size_t col, size_t row, int ci)
		{
			if (col < 0 || col >= _shp.height || row < 0 || row >= _shp.width || \
				ci < 0 || ci >= C)
				throw std::out_of_range(
					"[Error] `col` or `row` or `ci` out of range.");
			return _data[ci][col * _shp.width + row];
		}
	};

	// 图像类型
	// S|U|F --> 有符号整型|无符号整型|单精度浮点型
	// 1|3|4 --> 单通道图像|RGB彩色图像|RGBA带透明度彩色图像
	typedef Img<unsigned char, 1> ImgU8C1;
	typedef Img<unsigned char, 3> ImgU8C3;
	typedef Img<unsigned char, 4> ImgU8C4;
	typedef Img<unsigned int, 1> ImgS8C1;
	typedef Img<unsigned int, 3> ImgS8C3;
	typedef Img<unsigned int, 4> ImgS8C4;
	typedef Img<float, 1> ImgF8C1;
	typedef Img<float, 3> ImgF8C3;
	typedef Img<float, 4> ImgF8C4;
}
