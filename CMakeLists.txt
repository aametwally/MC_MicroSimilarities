cmake_minimum_required(VERSION 3.8)

include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (WIN32)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

elseif (UNIX)
    ##-------------------- Use settings at: https://cmake.org/Wiki/CMake_RPATH_handling#Always_full_RPATH
    # With these settings you will be able to
    # execute your programs from the build tree and
    # they will find the shared libraries in the build
    # tree and also the shared libraries outside your project,
    # when installing all executables and shared libraries
    # will be relinked, so they will find all libraries they need.
    ########################################################################
    ########################################################################
    # use, i.e. don't skip the full RPATH for the build tree
    SET(CMAKE_SKIP_BUILD_RPATH FALSE)

    # when building, don't use the install RPATH already
    # (but later on when installing)
    SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

    SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

    # add the automatically determined parts of the RPATH
    # which point to directories outside the build tree to the install RPATH
    SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


    # the RPATH to be used when installing, but only if it's not a system directory
    LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
    IF ("${isSystemDir}" STREQUAL "-1")
        SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    ENDIF ("${isSystemDir}" STREQUAL "-1")
    ########################################################################
    ########################################################################
endif ()


#add_dependencies(dlib openblas)
#add_dependencies(dlib lapack)
project(markovian_features C CXX)

### Add CMake modules
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

### Define test data directory and the output directory.
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

ExternalProject_Add(fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/fmt
        PREFIX ${EXTERNAL_INSTALL_LOCATION}
        CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
        -DCMAKE_VERBOSE_MAKEFILE=${CMAKE_VERBOSE_MAKEFILE}
        LOG_DOWNLOAD 1
        LOG_UPDATE 1
        LOG_INSTALL 1
        LOG_CONFIGURE 1
        LOG_BUILD 1)

#ExternalProject_Add(openblas
#        GIT_REPOSITORY "https://github.com/xianyi/OpenBLAS.git"
#        SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/openblas
#        PREFIX ${EXTERNAL_INSTALL_LOCATION}
#        CMAKE_ARGS
#        -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
#        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
#        -DCMAKE_VERBOSE_MAKEFILE=${CMAKE_VERBOSE_MAKEFILE}
#        BUILD_COMMAND make -j10
#        LOG_DOWNLOAD 1
#        LOG_UPDATE 1
#        LOG_INSTALL 1
#        LOG_CONFIGURE 1
#        LOG_BUILD 1
#        )
#
#ExternalProject_Add(lapack
#        GIT_REPOSITORY "https://github.com/Reference-LAPACK/lapack-release.git"
#        SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/lapack
#        PREFIX ${EXTERNAL_INSTALL_LOCATION}
#        CMAKE_ARGS
#        -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
#        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
#        -DCMAKE_VERBOSE_MAKEFILE=${CMAKE_VERBOSE_MAKEFILE}
#        BUILD_COMMAND make -j10
#        LOG_DOWNLOAD 1
#        LOG_UPDATE 1
#        LOG_INSTALL 1
#        LOG_CONFIGURE 1
#        LOG_BUILD 1 )


ExternalProject_Add(dlib
        GIT_REPOSITORY https://github.com/davisking/dlib
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/dlib
        PREFIX ${EXTERNAL_INSTALL_LOCATION}
        CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_VERBOSE_MAKEFILE=${CMAKE_VERBOSE_MAKEFILE}
        -DDLIB_USE_BLAS=OFF
        -DDLIB_USE_LAPACK=OFF
        BUILD_COMMAND make -j10
        LOG_DOWNLOAD 1
        LOG_UPDATE 1
        LOG_INSTALL 1
        LOG_CONFIGURE 1
        LOG_BUILD 1
        )


### Add external libraries
#message($ENV{FMT_SOURCE})
#message($ENV{DLIB_SOURCE})
#add_subdirectory($ENV{FMT_SOURCE} fmt_build )
#add_subdirectory($ENV{DLIB_SOURCE} dlib_build )

### Add all the files in the src directory to the CMake tree
add_subdirectory( src )


