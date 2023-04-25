#include <iostream>
#include <fstream>
#include <exception>
#include "bmp.h"

tcv::ImU8* tcv::bmp::LoadRGB(const char* fileName, Header* header)
{
	std::fstream ifs;
	ifs.open(fileName, std::ios::in | std::ios::binary);
	if (!ifs.is_open())
		throw std::invalid_argument("[Error] Can't open the file.");
	ifs.read((char*)&(header->file), sizeof(tcv::bmp::_File));
	if (header->file.bfType != tcv::bmp::BF_TYPE)
	{
		ifs.close();
		throw std::invalid_argument("[Error] Type error.");
	}
	ifs.read((char*)&(header->info), sizeof(tcv::bmp::_Info));
	// FIXME: 只能读取24位真彩色
	if (header->info.biBitCount != 24)
	{
		ifs.close();
		throw std::invalid_argument("[Error] Invalid biBitCount.");
	}
	int offset = (header->info.biWidth * 3) % 4;
	if (offset != 0)
		offset = 4 - offset;
	size_t fLen = size_t(header->info.biHeight) * size_t(header->info.biWidth);
	tcv::type::U8** surface = new tcv::type::U8 * [fLen];
	for (size_t i = 0; i < fLen; ++i)
		surface[i] = new tcv::type::U8[3];
	for (int i = header->info.biHeight - 1; i >= 0; --i)
	{
		for (int j = 0; j < header->info.biWidth; ++j)
			ifs.read(
				(char*)surface[i * size_t(header->info.biWidth) + j],
				sizeof(tcv::type::U8[3])
			);
		if (offset != 0)
		{
			char ign;
			for (int k = 0; k < offset; ++k)
				ifs.read(&ign, sizeof(char));
		}
	}
	// 新建im对象
	tcv::ImU8* img = new tcv::ImU8(
		size_t(header->info.biHeight),
		size_t(header->info.biWidth),
		3, surface, false
	);
	// 清理
	for (size_t i = 0; i < fLen; ++i)
		delete[] surface[i];
	delete[] surface;
	ifs.close();
	return img;
}

void tcv::bmp::saveRGB(
	const char* fileName, const tcv::ImU8* im, Header header)
{
	std::ofstream ofs;
	ofs.open(fileName, std::ios::out | std::ios::binary);
	ofs.write((char*)&header.file, sizeof(tcv::bmp::_File));
	ofs.write((char*)&header.info, sizeof(tcv::bmp::_Info));
	int offset = (header.info.biWidth * 3) % 4;
	if (offset != 0)
		offset = 4 - offset;
	for (int i = header.info.biHeight - 1; i >= 0; i--) 
	{
		for (int j = 0; j < header.info.biWidth; j++)
			ofs.write(
				(char*)(im->data(i, j, 0, 0)[0]),
				sizeof(tcv::type::U8[3])
			);
		if (offset != 0) 
		{
			char ign = 0;
			for (int k = 0; k < offset; k++)
				ofs.write(&ign, sizeof(char));
		}
	}
	ofs.close();
}
