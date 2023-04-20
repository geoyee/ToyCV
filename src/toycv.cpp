#if defined (WIN32)
	#define DLL_DEFINE
#endif

#include "toycv.h"

tcv::RGBTriple* tcv::imread(const char* fileName)
{
	tcv::BMP bmp;
	bmp.read(fileName);
	return bmp.array();
}
