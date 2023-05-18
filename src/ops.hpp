#pragma once

#include <iostream>
#include <exception>
#include "im.hpp"
#include "funcs.h"
#include "config.h"

#undef DLL_SPEC
#if defined(WIN32)
#if defined(DLL_DEFINE)
#define DLL_SPEC _declspec(dllexport)
#else
#define DLL_SPEC _declspec(dllimport)
#endif
#else
#define DLL_SPEC
#endif

namespace tcv
{
	namespace ops
	{
		template <typename T>
		DLL_SPEC tcv::Im<T>& RGB2GRAY(tcv::Im<T>& imgRGB)
		{
			tcv::Shape shp(imgRGB.shape());
			if (shp.channels != 3)
				throw std::logic_error("[Error] Channels must be 3.");
			shp.channels = 1;
			T** newData = _RGB2GRAY(imgRGB.data(), shp.height * shp.width);
			return *(new tcv::Im<T>(shp, newData));
		};
	}
}
