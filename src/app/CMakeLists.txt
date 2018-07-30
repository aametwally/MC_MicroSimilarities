## Dependencies
#include( Qt )
include_directories( ../algorithm )

### Generate the main executable
add_executable( markovian_features   markovian_features.cpp)
target_include_directories( markovian_features PRIVATE ${CMAKE_SOURCE_DIR}/src/include )
target_link_libraries( markovian_features PRIVATE fmt-header-only  )
target_compile_features( markovian_features PRIVATE cxx_std_17)

### Generate the subsetter executable
add_executable( subsetDataset   subsetDataset.cpp)
target_include_directories( subsetDataset PRIVATE ${CMAKE_SOURCE_DIR}/src/include )
target_link_libraries( subsetDataset PRIVATE fmt-header-only  )
target_compile_features( subsetDataset PRIVATE cxx_std_17)

# Install the executable (generate with 'make install' in the bin directory)
#install( TARGETS ${FINDING_CLUMPS_EXE} DESTINATION bin )
