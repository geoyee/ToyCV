﻿#pragma once

#include <stdio.h>
#include <stdlib.h>

// 只能直接在核函数使用
#define CHECK(call)                                                          \
	do                                                                       \
	{                                                                        \
		const cudaError_t error_code = call;                                 \
		if (error_code != cudaSuccess)                                       \
		{                                                                    \
			printf("CUDA Error:\n");                                         \
			printf("    File:        %s\n", __FILE__);                       \
			printf("    Line:        %d\n", __LINE__);                       \
			printf("    Error code:  %d\n", error_code);                     \
			printf("    Error text:  %s\n", cudaGetErrorString(error_code)); \
			system("pause");                                                 \
			exit(1);                                                         \
		}                                                                    \
	} while (0)
