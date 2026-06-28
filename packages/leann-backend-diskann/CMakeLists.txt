# Wrapper so USE_TCMALLOC is OFF before DiskANN's option() runs. DiskANN's
# FetchContent(tcmalloc) pulls protobuf/re2/fuzztest and fails on Linux CI when
# Abseil CMake targets are missing (e.g. ubuntu arm64 with apt protobuf only).
cmake_minimum_required(VERSION 3.24)
project(leann_backend_diskann_wrapper NONE)

set(USE_TCMALLOC OFF CACHE BOOL "Use tcmalloc from gperftools" FORCE)

if(APPLE)
    foreach(_leann_homebrew_prefix IN ITEMS
        /opt/homebrew/opt/protobuf
        /opt/homebrew/opt/abseil
        /opt/homebrew/opt/boost
        /opt/homebrew/opt/libomp
    )
        if(EXISTS "${_leann_homebrew_prefix}")
            list(PREPEND CMAKE_PREFIX_PATH "${_leann_homebrew_prefix}")
        endif()
    endforeach()

    if(EXISTS "/opt/homebrew/opt/boost" AND NOT DEFINED Boost_ROOT)
        set(Boost_ROOT "/opt/homebrew/opt/boost" CACHE PATH "Boost root" FORCE)
    endif()
    if(EXISTS "/opt/homebrew/opt/libomp" AND NOT DEFINED OpenMP_ROOT)
        set(OpenMP_ROOT "/opt/homebrew/opt/libomp" CACHE PATH "OpenMP root" FORCE)
    endif()
endif()

add_subdirectory(third_party/DiskANN)
