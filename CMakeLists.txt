cmake_minimum_required(VERSION 3.15)
project(jerasure C)

# gf-complete library path
set(GF_COMPLETE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../gf-complete")

# Jerasure/CMakeLists.txt
file(GLOB JERAS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)
add_library(jerasure STATIC ${JERAS_SOURCES})
target_include_directories(jerasure
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PUBLIC ${GF_COMPLETE_DIR}/include
)

# CRT Secure 警告抑制
target_compile_definitions(jerasure
    PRIVATE
        _CRT_SECURE_NO_WARNINGS
)

# gf-complete のヘッダー・ライブラリをリンク
add_subdirectory(${GF_COMPLETE_DIR} gf-complete)
target_link_libraries(jerasure PUBLIC gf_complete)
