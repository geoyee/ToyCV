#if defined (WIN32)
	#define DLL_DEFINE
#endif

#include "toycv.h"

char* tcv::readJPEG(const std::string& filePath)
{
	std::ifstream is(filePath, std::ifstream::in | std::ios::binary);
	is.seekg(0, is.end);
	int length = is.tellg();
	is.seekg(0, is.beg);
	char* buffer = new char[length];
	is.read(buffer, length);
	is.close();
	return buffer;
}
