#pragma once

#include <iostream>
#include <fstream>
#include "bmp.h"

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
	DLL_SPEC tcv::RGBTriple* imread(const char* fileName);
}
