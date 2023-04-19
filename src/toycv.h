#pragma once

#include <iostream>
#include <fstream>

#define Interface __declspec(dllexport)

namespace tcv {
	// 读取JPEG
	Interface char* readJPEG(const std::string& filePath);
}