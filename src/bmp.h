#pragma once

#include <iostream>

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
	struct DLL_SPEC RGBTriple
	{
		unsigned char blue;
		unsigned char green;
		unsigned char red;
	};

#pragma pack(2)
	struct DLL_SPEC BmpFileHeader
	{
		unsigned short bfType;
		unsigned int bfSize;
		unsigned short bfReserved1;
		unsigned short bfReserved2;
		unsigned int bfOffBits;
	};

	struct DLL_SPEC BmpFileInFoHeader
	{
		unsigned int biSize;
		int biWidth = 0, biHeight = 0;
		unsigned short biPlanes;
		unsigned short biBitCount;
		unsigned int biCompression, biSizeImages;
		int biXPelsPerMeter, biYPelsPerMeter;
		unsigned int biClrUsed, biClrImportant;
	};

	class DLL_SPEC BMP
	{
	private:
		const int TYPE = 19778;
		int offset;  // 行尾的空隙
		RGBTriple* surface;  // 存图片颜色数据的数组
		BmpFileHeader fileHeader;  // 文件头
		BmpFileInFoHeader fileInFoHeader;  // 数据头

	public:
		// 读文件
		void read(const char* fileName);
		// 写文件
		void write(void (*myMethod)(int, int, RGBTriple*), const char* outFileName);
		// 获取数据
		RGBTriple* array();
	};
}
