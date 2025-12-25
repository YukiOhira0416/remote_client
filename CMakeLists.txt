cmake_minimum_required(VERSION 3.20)
project(RemoteClient LANGUAGES CXX CUDA)

# C++20 標準を設定
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA 20 標準を設定
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Windowsプラットフォームの確認
if(NOT WIN32)
    message(FATAL_ERROR "This project is Windows only")
endif()

# デバッグ設定
set(CMAKE_BUILD_TYPE Debug)
#
# NOTE:
#   The previous configuration linked against the dynamic MSVC runtimes (/MD, /MDd).
#   Those DLLs (msvcp140d.dll, vcruntime140d.dll, etc.) are not guaranteed to exist on
#   end-user machines, which caused the executable to fail immediately at startup with
#   exit code 0xC0000135 (STATUS_DLL_NOT_FOUND).  Switching to the static runtime keeps
#   the required CRT inside the executable so the client can launch on systems without
#   a Visual Studio installation.
set(CMAKE_CXX_FLAGS_DEBUG "/MTd /Zi /Ob0 /Od /RTC1")
set(CMAKE_CXX_FLAGS_RELEASE "/MT /O2 /Ob2 /DNDEBUG")

# Unicode定義
add_definitions(-DUNICODE -D_UNICODE)

# プリプロセッサ定義
add_definitions(-DNOMINMAX -DWIN32_LEAN_AND_MEAN)

# インクルードディレクトリ
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/directx
    ${CMAKE_SOURCE_DIR}/concurrentqueue_x64-windows/include
    ${CMAKE_SOURCE_DIR}/enet_x64-windows/include
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include"
    "C:/Program Files/NVIDIA Corporation/Video_Codec_SDK_13.0.19/Interface"
)

# ISA-L サブモジュールをビルド
add_subdirectory(isa-l)

# ISA-L のインクルードディレクトリを設定
set(ISAL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/isa-l/include")
set(ISAL_LIBRARY "${CMAKE_SOURCE_DIR}/isa-l/isa-l_static.lib")

# プリコンパイル済みヘッダー対応のため必要
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# ソースファイル
set(SOURCES
    main.cpp
    window.cpp
    DebugLog.cpp
    Globals.cpp
    ReedSolomon.cpp
    nvdec.cpp
    AppShutdown.cpp
    TimeSyncClient.cpp
    AudioClient.cpp
)

# ヘッダーファイル
set(HEADERS
    main.h
    window.h
    DebugLog.h
    Globals.h
    ReedSolomon.h
    nvdec.h
    AppShutdown.h
    TimeSyncClient.h
    AudioClient.h
    AudioPacket.h
)

# 実行ファイルを作成
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# isa-l.dll をビルド後に実行ファイルと同じ場所へコピー
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/isa-l/isa-l.dll"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/isa-l.dll"
)

# ライブラリのリンク
target_link_libraries(${PROJECT_NAME}
    # Windows基本ライブラリ
    user32.lib
    ws2_32.lib
    Mswsock.lib
    ole32.lib
    uuid.lib
    bcrypt.lib
    Strmiids.lib
    winmm.lib
    secur32.lib
    
    # DirectX/D3D12ライブラリ
    d3d12.lib
    dxgi.lib
    d3dcompiler.lib
    
    # CUDA ライブラリ
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64/cuda.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64/cudart.lib"
    
    # NVDEC ライブラリ (Video Codec SDK)
    "C:/Program Files/NVIDIA Corporation/Video_Codec_SDK_13.0.19/Lib/x64/nvcuvid.lib"

    # ENetライブラリ
    ${CMAKE_SOURCE_DIR}/enet_x64-windows/lib/enet.lib

    # Intel ISA-L ライブラリ
    ${ISAL_LIBRARY}
    
    
)



# リンカーフラグ
set_target_properties(${PROJECT_NAME} PROPERTIES
    LINK_FLAGS "/SUBSYSTEM:WINDOWS"
)

# アプリケーションマニフェスト
if(EXISTS "${CMAKE_SOURCE_DIR}/app.manifest")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        LINK_FLAGS "/MANIFESTFILE:${CMAKE_SOURCE_DIR}/app.manifest"
    )
endif()

# デバッグ情報の生成
set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# プロパティの設定
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

# シェーダーコンパイル設定
find_program(FXC_COMPILER fxc.exe
    PATHS 
        "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64"
        "C:/Program Files (x86)/Windows Kits/10/bin/10.0.19041.0/x64"
        "C:/Program Files (x86)/Windows Kits/10/bin/*/x64"
        "C:/Program Files (x86)/Windows Kits/*/bin/*/x64"
    DOC "DirectX Shader Compiler"
)

# シェーダーファイルの設定
set(SHADER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/Shader")
set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/Shader")

# シェーダーファイルリスト
set(VERTEX_SHADERS
    "${SHADER_SOURCE_DIR}/FullScreenQuadVS.hlsl"
)

# 出力ディレクトリを作成
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

# シェーダーコンパイルの関数
function(compile_shader SHADER_FILE SHADER_TYPE OUTPUT_FILE)
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
    
    if(FXC_COMPILER)
        add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND ${FXC_COMPILER} /T ${SHADER_TYPE} /Fo ${OUTPUT_FILE} ${SHADER_FILE}
            DEPENDS ${SHADER_FILE}
            COMMENT "Compiling shader: ${SHADER_NAME}"
            VERBATIM
        )
    else()
        message(WARNING "DirectX Shader Compiler (fxc.exe) not found. Skipping shader compilation.")
    endif()
endfunction()

# バーテックスシェーダーのコンパイル
set(COMPILED_VERTEX_SHADERS)
foreach(SHADER IN LISTS VERTEX_SHADERS)
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(OUTPUT_FILE "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.cso")
    compile_shader(${SHADER} vs_5_0 ${OUTPUT_FILE})
    list(APPEND COMPILED_VERTEX_SHADERS ${OUTPUT_FILE})
endforeach()

# ピクセルシェーダーのコンパイル
set(COMPILED_PIXEL_SHADERS)
foreach(SHADER IN LISTS PIXEL_SHADERS)
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(OUTPUT_FILE "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.cso")
    compile_shader(${SHADER} ps_5_0 ${OUTPUT_FILE})
    list(APPEND COMPILED_PIXEL_SHADERS ${OUTPUT_FILE})
endforeach()

# シェーダーコンパイル用のカスタムターゲット
add_custom_target(CompileShaders ALL
    DEPENDS ${COMPILED_VERTEX_SHADERS} ${COMPILED_PIXEL_SHADERS}
    COMMENT "Compiling all shaders"
)

# メインプロジェクトがシェーダーに依存するよう設定
add_dependencies(${PROJECT_NAME} CompileShaders)

# 実行時に必要なファイルをコピー
if(EXISTS "${CMAKE_SOURCE_DIR}/enet_x64-windows/lib/enet.dll")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/enet_x64-windows/lib/enet.dll"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif()

# --- Shaderフォルダを実行ファイルと同じ場所にコピー（最低限） ---
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/Shader"
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/Shader
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/isa-l/isa-l.dll"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/isa-l.dll"
    COMMENT "Copying isa-l.dll to output directory"
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/isa-l/isa-l.lib"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/isa-l.lib"
    COMMENT "Copying isa-l.lib to output directory"
)