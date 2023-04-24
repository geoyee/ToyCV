#include <iostream>
#include <fstream>
#include <exception>
#include "bmp.h"

tcv::ImU8* tcv::bmp::readRGB(const char* fileName)
{
	int offset;                          // 行尾的空隙
	bmp::FileHeader fileHeader;          // 文件头
	bmp::FileInFoHeader fileInFoHeader;  // 数据头
	std::fstream ifs;
	ifs.open(fileName, std::ios::in | std::ios::binary);
	if (!ifs.is_open())
		throw std::invalid_argument("Can't open the file.");
	ifs.read((char*)&fileHeader, sizeof(tcv::bmp::FileHeader));
	if (fileHeader.bfType != tcv::bmp::TYPE)
		throw std::invalid_argument("Type error.");
	ifs.read((char*)&fileInFoHeader, sizeof(tcv::bmp::FileInFoHeader));
	if (fileInFoHeader.biBitCount != 24)
		throw std::invalid_argument("Invalid biBitCount.");
	offset = (fileInFoHeader.biWidth * 3) % 4;
	if (offset != 0)
		offset = 4 - offset;
	size_t fLen = size_t(fileInFoHeader.biHeight) * size_t(fileInFoHeader.biWidth);
	tcv::type::U8** surface = new tcv::type::U8 * [fLen];
	for (size_t i = 0; i < fLen; ++i)
		surface[i] = new tcv::type::U8[3];
	for (int i = fileInFoHeader.biHeight - 1; i >= 0; --i)
	{
		for (int j = 0; j < fileInFoHeader.biWidth; ++j)
			ifs.read(
				(char*)surface[i * size_t(fileInFoHeader.biWidth) + j],
				sizeof(tcv::type::U8[3])
			);
		if (offset != 0)
		{
			char ign;
			for (int k = 0; k < offset; ++k)
				ifs.read(&ign, sizeof(char));
		}
	}
	ImU8* img = new ImU8(
		size_t(fileInFoHeader.biHeight),
		size_t(fileInFoHeader.biWidth),
		3, surface, false);
	for (size_t i = 0; i < fLen; ++i)
		delete[] surface[i];
	delete[] surface;
	ifs.close();
	return img;
}
