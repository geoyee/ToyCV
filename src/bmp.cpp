#if defined (WIN32)
#define DLL_DEFINE
#endif

#include <iostream>
#include <fstream>
#include "bmp.h"

void tcv::BMP::read(const char* fileName)
{
	std::fstream ifs;
	ifs.open(fileName, std::ios::in | std::ios::binary);
	if (!ifs.is_open())
	{
		std::cout << "Can't open the file." << std::endl;
		return;
	}
	ifs.read((char*)&fileHeader, sizeof(BmpFileHeader));
	if (fileHeader.bfType != BMP::TYPE)
	{
		std::cout << "Type error, " << fileHeader.bfType
			<< " != " << BMP::TYPE << std::endl;
		return;
	}
	ifs.read((char*)&fileInFoHeader, sizeof(BmpFileInFoHeader));
	if (fileInFoHeader.biBitCount != 24)
	{
		std::cout << "Invalid, " << fileInFoHeader.biBitCount
			<< " != 24" << std::endl;
		return;
	}
	offset = (fileInFoHeader.biWidth * 3) % 4;
	if (offset != 0)
		offset = 4 - offset;
	surface = new RGBTriple[
		size_t(fileInFoHeader.biHeight) * size_t(fileInFoHeader.biWidth)];
	for (int i = fileInFoHeader.biHeight - 1; i >= 0; --i)
	{
		for (int j = 0; j < fileInFoHeader.biWidth; ++j)
			ifs.read(
				(char*)(surface + (size_t(fileInFoHeader.biWidth) * i + j)),
				sizeof(RGBTriple)
			);
		if (offset != 0)
		{
			char ign;
			for (int k = 0; k < offset; ++k)
				ifs.read(&ign, sizeof(char));
		}
	}
	ifs.close();
}

void tcv::BMP::write(void (*preMethod)(int, int, RGBTriple*), const char* outFileName)
{
	preMethod(fileInFoHeader.biHeight, fileInFoHeader.biWidth, surface);
	std::ofstream ofs;
	ofs.open(outFileName, std::ios::out | std::ios::binary);
	ofs.write((char*)&fileHeader, sizeof(BmpFileHeader));
	ofs.write((char*)&fileInFoHeader, sizeof(BmpFileInFoHeader));
	for (int i = fileInFoHeader.biHeight - 1; i >= 0; --i)
	{
		for (int j = 0; j < fileInFoHeader.biWidth; ++j)
			ofs.write(
				(char*)(surface + (size_t(fileInFoHeader.biWidth) * i + j)),
				sizeof(RGBTriple)
			);
		if (offset != 0)
		{
			char ign = 0;
			for (int k = 0; k < offset; ++k)
				ofs.write(&ign, sizeof(char));
		}
	}
	delete[] surface;
	ofs.close();
}

tcv::RGBTriple* tcv::BMP::array()
{
	return surface;
}
