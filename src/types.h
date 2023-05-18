#pragma once

namespace tcv
{
	// 常用图像类型
	namespace type
	{
		typedef unsigned char U8;		// [0             , 255          )
		typedef signed char S8;			// [-128          , 127          )
		typedef unsigned short int U16; // [0             , 65535        )
		typedef signed short int S16;	// [-32768        , 32767        )
		typedef signed int S32;			// [-2147483648   , 2147483647   )
		typedef float F32;				// [1.18*10^{-38} , 3.40*10^{38} )
		typedef double F64;				// [2.23*10^{-308}, 1.79*10^{308})
	}
}
