cmake_minimum_required(VERSION 3.4 )

if(WIN32)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

elseif(UNIX)
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
    SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

    # when building, don't use the install RPATH already
    # (but later on when installing)
    SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

    SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

    # add the automatically determined parts of the RPATH
    # which point to directories outside the build tree to the install RPATH
    SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


    # the RPATH to be used when installing, but only if it's not a system directory
    LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
    IF("${isSystemDir}" STREQUAL "-1")
       SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    ENDIF("${isSystemDir}" STREQUAL "-1")
    ########################################################################
    ########################################################################
endif()

project( markovian_features C CXX )

### Add CMake modules
set( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )

### Define test data directory and the output directory.
set( TEST_DATA_PATH ${CMAKE_SOURCE_DIR}/data)
set( EXPECTED_TEST_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/output)


### Add external libraries
add_subdirectory($ENV{FMT_SOURCE} fmt_build )
add_subdirectory($ENV{DLIB_SOURCE} dlib_build )

### Add all the files in the src directory to the CMake tree
add_subdirectory( src )

