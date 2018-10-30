set( SEGMENTATION_SOURCES
        SequenceSegmentation.cpp )


set( SEGMENTATION_INCLUDES
        SequenceSegmentation.hpp )

set( SEGMENTATION_LIB segmentation_lib )

add_library( ${SEGMENTATION_LIB} STATIC ${SEGMENTATION_INCLUDES} ${SEGMENTATION_SOURCES} )
target_include_directories( ${SEGMENTATION_LIB} INTERFACE ${CMAKE_SOURCE_DIR}/src/include  )

target_link_libraries( ${SEGMENTATION_LIB} INTERFACE fmt-header-only stdc++fs )
target_compile_features( ${SEGMENTATION_LIB} INTERFACE cxx_std_17)

set( SEGMENTATION_LIB ${SEGMENTATION_LIB} PARENT_SCOPE )
