﻿#pragma once

#include "funcs.h"
#include "config.h"

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
	namespace funcs
	{
		template<typename T>
		DLL_SPEC void addNumberInplace(T** data, size_t fLen, size_t C, double val)
		{
			_addNumberInplace(data, fLen, C, val);
			// FIXME: 没有同步
			for (int i = 0; i < 10; ++i)
				printf("pass new val: %d\n", data[0][i]);
		};
	}
}