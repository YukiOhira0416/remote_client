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
set(CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /Ob0 /Od /RTC1")
set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Ob2 /DNDEBUG")

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
    ${CMAKE_SOURCE_DIR}/gf-complete/include
    ${CMAKE_SOURCE_DIR}/Jerasure/include
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include"
    "C:/Program Files/NVIDIA Corporation/Video_Codec_SDK_13.0.19/Interface"
)

# Jerasure and gf-complete libraries
set(GF_COMPLETE_LIB "${CMAKE_SOURCE_DIR}/gf-complete/build/Debug/gf_complete.lib")
set(JERASURE_LIB "${CMAKE_SOURCE_DIR}/Jerasure/build/Debug/jerasure.lib")



# プリコンパイル済みヘッダー対応のため必要
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# ソースファイル
set(SOURCES
    main.cpp
    window.cpp
    DebugLog.cpp
    Globals.cpp
    ReedSolomon.cpp
    Nvdec.cpp
)

# ヘッダーファイル
set(HEADERS
    main.h
    window.h
    DebugLog.h
    Globals.h
    ReedSolomon.h
    Nvdec.h
)

# 実行ファイルを作成
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

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
    
    # MediaFoundationライブラリ
    mfplat.lib
    mfuuid.lib
    mfreadwrite.lib
    
    # CUDA ライブラリ
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64/cuda.lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64/cudart.lib"
    
    # NVDEC ライブラリ (Video Codec SDK)
    "C:/Program Files/NVIDIA Corporation/Video_Codec_SDK_13.0.19/Lib/x64/nvcuvid.lib"
    
    # ENetライブラリ
    ${CMAKE_SOURCE_DIR}/enet_x64-windows/lib/enet.lib
    
    # Jerasure and gf-complete libraries
    ${JERASURE_LIB}
    ${GF_COMPLETE_LIB}
    
    
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

set(PIXEL_SHADERS
    "${SHADER_SOURCE_DIR}/NV12ToRGBPS.hlsl"
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

# VS Code設定用のフォルダ作成とファイル設置
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/.vscode")

# launch.json用の設定
set(LAUNCH_JSON_CONTENT "{
    \"version\": \"0.2.0\",
    \"configurations\": [
        {
            \"name\": \"(Windows) 起動\",
            \"type\": \"cppvsdbg\",
            \"request\": \"launch\",
            \"program\": \"\${workspaceFolder}/build/Debug/RemoteClient.exe\",
            \"args\": [],
            \"stopAtEntry\": false,
            \"cwd\": \"\${workspaceFolder}\",
            \"environment\": [],
            \"console\": \"externalTerminal\",
            \"preLaunchTask\": \"build\"
        }
    ]
}")

file(WRITE "${CMAKE_SOURCE_DIR}/.vscode/launch.json" "${LAUNCH_JSON_CONTENT}")

# tasks.json用の設定
set(TASKS_JSON_CONTENT "{
    \"version\": \"2.0.0\",
    \"tasks\": [
        {
            \"label\": \"build\",
            \"type\": \"shell\",
            \"command\": \"cmake\",
            \"args\": [
                \"--build\",
                \"\${workspaceFolder}/build\",
                \"--config\",
                \"Debug\"
            ],
            \"group\": {
                \"kind\": \"build\",
                \"isDefault\": true
            },
            \"presentation\": {
                \"echo\": true,
                \"reveal\": \"always\",
                \"focus\": false,
                \"panel\": \"shared\",
                \"showReuseMessage\": true,
                \"clear\": false
            },
            \"problemMatcher\": \"\$msCompile\"
        },
        {
            \"label\": \"configure\",
            \"type\": \"shell\",
            \"command\": \"cmake\",
            \"args\": [
                \"-S\",
                \"\${workspaceFolder}\",
                \"-B\",
                \"\${workspaceFolder}/build\",
                \"-G\",
                \"Visual Studio 17 2022\"
            ],
            \"group\": \"build\",
            \"presentation\": {
                \"echo\": true,
                \"reveal\": \"always\",
                \"focus\": false,
                \"panel\": \"shared\",
                \"showReuseMessage\": true,
                \"clear\": false
            }
        },
        {
            \"label\": \"clean\",
            \"type\": \"shell\",
            \"command\": \"cmake\",
            \"args\": [
                \"--build\",
                \"\${workspaceFolder}/build\",
                \"--target\",
                \"clean\"
            ],
            \"group\": \"build\"
        }
    ]
}")

file(WRITE "${CMAKE_SOURCE_DIR}/.vscode/tasks.json" "${TASKS_JSON_CONTENT}")

# c_cpp_properties.json用の設定
set(CPP_PROPERTIES_JSON_CONTENT "{
    \"version\": 4,
    \"configurations\": [
        {
            \"name\": \"Win32\",
            \"includePath\": [
                \"\${workspaceFolder}/**\",
                \"\${workspaceFolder}/directx\",
                \"\${workspaceFolder}/concurrentqueue_x64-windows/include\",
                \"\${workspaceFolder}/enet_x64-windows/include\",
                \"\${workspaceFolder}/gf-complete/include\",
                \"\${workspaceFolder}/Jerasure/include\",
                \"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include\",
                \"C:/Program Files/NVIDIA Corporation/Video_Codec_SDK_13.0.19/Interface\"
            ],
            \"defines\": [
                \"_DEBUG\",
                \"UNICODE\",
                \"_UNICODE\",
                \"NOMINMAX\",
                \"WIN32_LEAN_AND_MEAN\"
            ],
            \"windowsSdkVersion\": \"10.0.22000.0\",
            \"compilerPath\": \"C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.40.33807/bin/Hostx64/x64/cl.exe\",
            \"cStandard\": \"c17\",
            \"cppStandard\": \"c++20\",
            \"intelliSenseMode\": \"windows-msvc-x64\",
            \"configurationProvider\": \"ms-vscode.cmake-tools\"
        }
    ]
}")

file(WRITE "${CMAKE_SOURCE_DIR}/.vscode/c_cpp_properties.json" "${CPP_PROPERTIES_JSON_CONTENT}")
