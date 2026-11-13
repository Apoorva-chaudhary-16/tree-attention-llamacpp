cmake_minimum_required(VERSION 3.14)
project(op_profile CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Links straight against the llama.cpp BUILD TREE made by scripts/00_build_llamacpp.py
# (no `make install` needed).  Pass -DLLAMA_SRC_DIR=... -DLLAMA_BUILD_DIR=...
if (NOT LLAMA_SRC_DIR OR NOT LLAMA_BUILD_DIR)
  message(FATAL_ERROR "pass -DLLAMA_SRC_DIR=<llama.cpp checkout> -DLLAMA_BUILD_DIR=<its build dir>")
endif()

set(_LIBDIRS
  ${LLAMA_BUILD_DIR}/bin ${LLAMA_BUILD_DIR}/bin/Release
  ${LLAMA_BUILD_DIR}/src ${LLAMA_BUILD_DIR}/src/Release
  ${LLAMA_BUILD_DIR}/ggml/src ${LLAMA_BUILD_DIR}/ggml/src/Release)

function(need_lib var name)
  find_library(${var} NAMES ${name} PATHS ${_LIBDIRS} NO_DEFAULT_PATH)
  if (NOT ${var})
    message(FATAL_ERROR "library '${name}' not found under ${LLAMA_BUILD_DIR}")
  endif()
endfunction()

need_lib(LIB_LLAMA llama)
need_lib(LIB_GGML ggml)
need_lib(LIB_GGML_BASE ggml-base)
find_library(LIB_GGML_CPU NAMES ggml-cpu PATHS ${_LIBDIRS} NO_DEFAULT_PATH)   # absent in static builds

add_executable(op_profile op_profile.cpp)
target_include_directories(op_profile PRIVATE ${LLAMA_SRC_DIR}/include ${LLAMA_SRC_DIR}/ggml/include)
target_link_libraries(op_profile PRIVATE ${LIB_LLAMA} ${LIB_GGML} ${LIB_GGML_BASE} ${LIB_GGML_CPU})
set_target_properties(op_profile PROPERTIES BUILD_RPATH "${LLAMA_BUILD_DIR}/bin")
if (WIN32)   # DLLs must sit next to the exe on Windows
  add_custom_command(TARGET op_profile POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${LLAMA_BUILD_DIR}/bin/Release $<TARGET_FILE_DIR:op_profile>)
endif()
