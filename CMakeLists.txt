
cmake_minimum_required(VERSION 3.18)
project(grca LANGUAGES C CXX)

# Vulkan CSV Viewer
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
find_package(Vulkan REQUIRED)
add_executable(csv_viewer cpp/csv_viewer.cpp cpp/vk_common.cpp)
add_executable(compare_csv cpp/compare_csv.cpp)
target_compile_options(compare_csv PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-march=native;-ffast-math;-funroll-loops>
  $<$<CXX_COMPILER_ID:MSVC>:/O2;/fp:fast>
)
target_include_directories(csv_viewer PRIVATE ${GLFW_INCLUDE_DIRS})
target_link_libraries(csv_viewer Vulkan::Vulkan glfw)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# =============================================================================
# nlohmann/json  (bundled in third_party/nlohmann/json.hpp)
# =============================================================================

# =============================================================================
# Detect CUDA
# =============================================================================
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(HAVE_CUDA TRUE)
  message(STATUS "CUDA compiler : ${CMAKE_CUDA_COMPILER}")
else()
  set(HAVE_CUDA FALSE)
  message(STATUS "CUDA not found — CPU-only build")
endif()

# =============================================================================
# Detect OptiX SDK
# =============================================================================
set(HAVE_OPTIX FALSE)
if(HAVE_CUDA)
  set(OptiX_INSTALL_DIR "" CACHE PATH
    "OptiX SDK root  (e.g. /usr/local/optix or ~/NVIDIA-OptiX-SDK-8.0.0-linux64)")
  find_path(OPTIX_INCLUDE_DIR
    NAMES optix.h
    PATHS
      "${OptiX_INSTALL_DIR}/include"
      "$ENV{OptiX_INSTALL_DIR}/include"
      "/usr/local/optix/include"
      "/opt/optix/include"
  )
  if(OPTIX_INCLUDE_DIR)
    set(HAVE_OPTIX TRUE)
    message(STATUS "OptiX headers : ${OPTIX_INCLUDE_DIR}")
  else()
    message(STATUS "OptiX not found — pass -DOptiX_INSTALL_DIR=<path> to enable")
  endif()
endif()



# =============================================================================
# Detect Embree 4.4
# =============================================================================
set(HAVE_EMBREE FALSE)
find_package(embree 4.4 CONFIG QUIET)
if(embree_FOUND)
  set(HAVE_EMBREE TRUE)
  message(STATUS "Embree 4     : found via CMake config")
else()
  message(STATUS "Embree 4.4 not found — build from source (see README) to enable the embree backend")
endif()

# =============================================================================
# GLSL shader compilation (best-effort; Makefile is the authoritative builder)
# =============================================================================
find_program(GLSLC glslc
  HINTS "${Vulkan_INCLUDE_DIRS}/../bin" "$ENV{VULKAN_SDK}/bin"
  PATHS /usr/bin /usr/local/bin
)
find_program(GLSLANGVALIDATOR glslangValidator
  HINTS "${Vulkan_INCLUDE_DIRS}/../bin" "$ENV{VULKAN_SDK}/bin"
  PATHS /usr/bin /usr/local/bin
)
if(GLSLC)
  set(_SHADER_COMPILER ${GLSLC})
  set(_SHADER_FLAGS "")
elseif(GLSLANGVALIDATOR)
  set(_SHADER_COMPILER ${GLSLANGVALIDATOR})
  set(_SHADER_FLAGS -V)
endif()
if(_SHADER_COMPILER)
  set(_shader_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
  set(_shader_build_dir "${CMAKE_CURRENT_SOURCE_DIR}/build")
  add_custom_target(shaders ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_shader_build_dir}"
    COMMAND ${_SHADER_COMPILER} ${_SHADER_FLAGS} "${_shader_src_dir}/hit_point.vert" -o "${_shader_build_dir}/hit_point.vert.spv"
    COMMAND ${_SHADER_COMPILER} ${_SHADER_FLAGS} "${_shader_src_dir}/hit_point.frag" -o "${_shader_build_dir}/hit_point.frag.spv"
    COMMAND ${_SHADER_COMPILER} ${_SHADER_FLAGS} "${_shader_src_dir}/mesh_wire.vert" -o "${_shader_build_dir}/mesh_wire.vert.spv"
    COMMAND ${_SHADER_COMPILER} ${_SHADER_FLAGS} "${_shader_src_dir}/mesh_wire.frag" -o "${_shader_build_dir}/mesh_wire.frag.spv"
    COMMAND ${_SHADER_COMPILER} ${_SHADER_FLAGS} "${_shader_src_dir}/cone.frag" -o "${_shader_build_dir}/cone.frag.spv"
    COMMENT "Compiling GLSL shaders to SPIR-V in build/"
    VERBATIM
  )
else()
  message(STATUS "glslc/glslangValidator not found — run 'make shaders' or pre-compile .spv files manually")
endif()

# =============================================================================
# Detect TinyBVH (bundled in third_party/tiny_bvh.h — always available)
# =============================================================================
set(HAVE_TINYBVH FALSE)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tiny_bvh.h")
  set(HAVE_TINYBVH TRUE)
  message(STATUS "TinyBVH     : found (third_party/tiny_bvh.h)")
else()
  message(STATUS "TinyBVH     : not found — run 'make' to fetch tiny_bvh.h")
endif()

# =============================================================================
# Sources
# =============================================================================

# =============================================================================
# Sources (no duplicates, all explicit)
# =============================================================================
set(GRCA_SOURCES
  main.cpp
  cpp/GRCA_CPU.cpp
  cpp/hit_viewer.cpp
  cpp/vk_common.cpp
)

if(HAVE_CUDA)
  list(APPEND GRCA_SOURCES cuda/GRCA_CUDA.cu)
endif()

if(HAVE_OPTIX)
  list(APPEND GRCA_SOURCES optix_cuda/optix.cu)
  if(HAVE_CUDA)
    list(APPEND GRCA_SOURCES cpp/hybrid_gpu.cpp)
  endif()
endif()

if(HAVE_EMBREE)
  list(APPEND GRCA_SOURCES cpp/embree.cpp)
  list(APPEND GRCA_SOURCES cpp/hybrid_cpu.cpp)
endif()

if(HAVE_TINYBVH)
  list(APPEND GRCA_SOURCES cpp/tinybvh_backends.cpp)
  if(HAVE_CUDA)
    list(APPEND GRCA_SOURCES cuda/tinybvh_kernels.cu)
  endif()
endif()

# =============================================================================
# Main executable
# =============================================================================
add_executable(grca ${GRCA_SOURCES})

if(_SHADER_COMPILER)
  add_dependencies(grca shaders)
endif()

target_include_directories(grca PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party"   # nlohmann/  → <nlohmann/json.hpp>, tiny_obj_loader.h
)

target_compile_definitions(grca PRIVATE
  $<$<BOOL:${HAVE_CUDA}>:HAVE_CUDA>
  $<$<BOOL:${HAVE_OPTIX}>:HAVE_OPTIX>
  $<$<BOOL:${HAVE_EMBREE}>:HAVE_EMBREE>
  $<$<BOOL:${HAVE_TINYBVH}>:HAVE_TINYBVH>
)

# C++ compile options
target_compile_options(grca PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-march=native;-ffast-math;-funroll-loops;-Wall;-Wextra>>
  $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/O2;/fp:fast;/W3>>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3>
)

if(HAVE_CUDA)
  # Determine CUDA architecture once; reuse for grca and grca_simd.
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set(_GRCA_CUDA_ARCH native)
  else()
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
      execute_process(
        COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE _gpu_cap
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _smi_result
      )
      if(_smi_result EQUAL 0 AND _gpu_cap MATCHES "^[0-9]+\\.[0-9]+$")
        string(REPLACE "." "" _GRCA_CUDA_ARCH "${_gpu_cap}")
      else()
        message(STATUS "nvidia-smi returned unexpected output — defaulting CUDA_ARCHITECTURES to 86")
        set(_GRCA_CUDA_ARCH 86)
      endif()
    else()
      set(_GRCA_CUDA_ARCH 86)
    endif()
  endif()

  set_property(TARGET grca PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET grca PROPERTY CUDA_ARCHITECTURES ${_GRCA_CUDA_ARCH})
endif()


if(HAVE_OPTIX)
  target_include_directories(grca PRIVATE "${OPTIX_INCLUDE_DIR}")
  # OptiX uses the CUDA driver API (cuCtxGetCurrent etc.) — link the driver stub
  target_link_libraries(grca PRIVATE CUDA::cuda_driver)
endif()

if(HAVE_EMBREE)
  if(embree_FOUND)
    target_link_libraries(grca PRIVATE embree)
  else()
    target_include_directories(grca PRIVATE "${EMBREE_INCLUDE_DIR}")
    target_link_libraries(grca PRIVATE "${EMBREE_LIBRARY}")
  endif()
endif()

# Always link Vulkan and GLFW for grca (needed for hit_viewer)
target_link_libraries(grca PRIVATE Vulkan::Vulkan glfw)

# =============================================================================
# OptiX device PTX  (compiled separately, deployed next to the binary)
# =============================================================================
if(HAVE_OPTIX)
  add_library(optix_device OBJECT optix_cuda/optix_device.cu)
  set_property(TARGET optix_device PROPERTY CUDA_PTX_COMPILATION ON)
  set_property(TARGET optix_device PROPERTY CUDA_ARCHITECTURES 75)
  target_compile_options(optix_device PRIVATE --use_fast_math;-O3;--diag-suppress=20044)
  target_include_directories(optix_device PRIVATE
    "${OPTIX_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/optix_cuda"
  )
  add_custom_command(TARGET grca POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_OBJECTS:optix_device>
      $<TARGET_FILE_DIR:grca>/optix_device.ptx
    COMMENT "Deploying optix_device.ptx alongside grca binary"
  )
  add_dependencies(grca optix_device)

  # Custom-IS PTX (software Möller–Trumbore, no RT-core triangle intersection)
  add_library(optix_custom_device OBJECT optix_cuda/optix_custom_device.cu)
  set_property(TARGET optix_custom_device PROPERTY CUDA_PTX_COMPILATION ON)
  set_property(TARGET optix_custom_device PROPERTY CUDA_ARCHITECTURES 75)
  target_compile_options(optix_custom_device PRIVATE --use_fast_math;-O3;--diag-suppress=20044)
  target_include_directories(optix_custom_device PRIVATE
    "${OPTIX_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/optix_cuda"
  )
  add_custom_command(TARGET grca POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_OBJECTS:optix_custom_device>
      $<TARGET_FILE_DIR:grca>/optix_custom_device.ptx
    COMMENT "Deploying optix_custom_device.ptx alongside grca binary"
  )
  add_dependencies(grca optix_custom_device)
endif()

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== GRCA build configuration ===")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CPU backend: ON (always)")
message(STATUS "  CUDA backend: ${HAVE_CUDA}")
message(STATUS "  OptiX backend: ${HAVE_OPTIX}")
message(STATUS "  Embree backend: ${HAVE_EMBREE}")
message(STATUS "  TinyBVH CPU+GPU backend: ${HAVE_TINYBVH}")
message(STATUS "  Hybrid-CPU backend (Embree+GRCA): ${HAVE_EMBREE}")
if(HAVE_OPTIX AND HAVE_CUDA)
  message(STATUS "  Hybrid-GPU backend (OptiX+GRCA): ON")
else()
  message(STATUS "  Hybrid-GPU backend (OptiX+GRCA): OFF (requires CUDA + OptiX)")
endif()
message(STATUS "  Visualiser (GLFW): ON (always)")
message(STATUS "================================")
message(STATUS "")
