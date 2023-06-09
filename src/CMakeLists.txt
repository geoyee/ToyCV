﻿SET(PORJ_NAME toycv)

# 配置一个头文件，通过该文件向源码传递CMake设置
CONFIGURE_FILE(
	"${PROJECT_SOURCE_DIR}/config.h.in"
	"${PROJECT_BINARY_DIR}/config.h"
)

# 将二进制的文件树包含到文件的搜索路径中，这样才能找到ProjrctConfig.h
INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR})

SET(LIB_SRC bmp.cpp funcs.cu)
SET(LIB_HEAD types.h bmp.h funcs.h im.hpp ops.hpp error.cuh)

SET(OBJ_NAME ${PORJ_NAME}_obj)
ADD_LIBRARY(${OBJ_NAME} OBJECT ${LIB_SRC})
# 构建动态库
ADD_LIBRARY(${PORJ_NAME} SHARED $<TARGET_OBJECTS:${OBJ_NAME}>)
# 构建静态库
SET(STATIC_NAME ${PORJ_NAME}_static)
ADD_LIBRARY(${STATIC_NAME} STATIC $<TARGET_OBJECTS:${OBJ_NAME}>)
# SET_TARGET_PROPERTIES(${STATIC_NAME} PROPERTIES OUTPUT_NAME ${PORJ_NAME})

# 编译为位置独立的代码
SET_SOURCE_FILES_PROPERTIES(${LIB_SRC} PROPERTIES COMPILE_FLAGS "-fPIC")

# 安装
INSTALL(TARGETS ${PORJ_NAME} ${STATIC_NAME}
	RUNTIME DESTINATION lib
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
)
INSTALL(FILES ${LIB_HEAD} DESTINATION "include/toycv")
INSTALL(FILES "${PROJECT_BINARY_DIR}/config.h" DESTINATION "include/toycv")
