#pragma once

#include <iostream>
#include <fstream>

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

namespace tcv {
	// 读取JPEG
	DLL_SPEC char* readJPEG(const std::string& filePath);
}
