#pragma once

#include <iostream>
#include "im.hpp"

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
	namespace bmp
	{
		const int TYPE = 19778;

#pragma pack(2)
		struct DLL_SPEC FileHeader
		{
			unsigned short bfType = 0;
			unsigned int bfSize = 0;
			unsigned short bfReserved1 = 0;
			unsigned short bfReserved2 = 0;
			unsigned int bfOffBits = 0;
		};

		struct DLL_SPEC FileInFoHeader
		{
			unsigned int biSize = 0;
			int biWidth = 0, biHeight = 0;
			unsigned short biPlanes = 0;
			unsigned short biBitCount = 0;
			unsigned int biCompression = 0, biSizeImages = 0;
			int biXPelsPerMeter = 0, biYPelsPerMeter = 0;
			unsigned int biClrUsed = 0, biClrImportant = 0;
		};

		DLL_SPEC tcv::ImU8* readRGB(const char* fileName);
	}
}
