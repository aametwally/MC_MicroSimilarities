set( SEGMENTATION_SOURCES
        SequenceAnnotator.cpp)


set( SEGMENTATION_INCLUDES
        SequenceAnnotator.hpp )

set( SEGMENTATION_LIB segmentation_lib )

add_library( ${SEGMENTATION_LIB} STATIC ${SEGMENTATION_INCLUDES} ${SEGMENTATION_SOURCES} )

target_include_directories( ${SEGMENTATION_LIB} PUBLIC .
        ${CMAKE_SOURCE_DIR}/src/include
        ${EXTERNAL_INSTALL_LOCATION}/include )
target_link_libraries( ${SEGMENTATION_LIB} PUBLIC CONAN_PKG::fmt stdc++fs )
target_compile_features( ${SEGMENTATION_LIB} PUBLIC cxx_std_17)
