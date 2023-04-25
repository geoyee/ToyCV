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
		const int BF_TYPE = 0x4D42;

#pragma pack(2)
		struct DLL_SPEC _File
		{
			unsigned short bfType = 19778;
			unsigned int bfSize = 0;
			unsigned short bfReserved1 = 0;
			unsigned short bfReserved2 = 0;
			unsigned int bfOffBits = 0;
		};

		struct DLL_SPEC _Info
		{
			unsigned int biSize = 40;
			int biWidth = 0;
			int biHeight = 0;
			unsigned short biPlanes = 1;
			unsigned short biBitCount = 24;
			unsigned int biCompression = 0;
			unsigned int biSizeImage = 0;
			int biXPelsPerMeter = 0;
			int	biYPelsPerMeter = 0;
			unsigned int biClrUsed = 0;
			unsigned int biClrImportant = 0;
		};

		struct DLL_SPEC Header
		{
			_File file;
			_Info info;
		};

		DLL_SPEC tcv::ImU8* LoadRGB(const char* fileName, Header* header);
		DLL_SPEC void saveRGB(const char* fileName, const tcv::ImU8* im, Header header);
	}
}
