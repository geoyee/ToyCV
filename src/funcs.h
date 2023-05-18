#pragma once

#include "config.h"
#include "types.h"

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

template <typename T>
DLL_SPEC T** _RGB2GRAY(T** data, size_t fLen);
