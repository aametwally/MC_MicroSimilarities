## Dependencies
#include( Qt )
include_directories( ../algorithm )

### Generate the executable
add_executable( markovian_features   markovian_features.cpp)
target_link_libraries( markovian_features PRIVATE fmt-header-only )
target_compile_features( markovian_features PRIVATE cxx_std_17)
# Install the executable (generate with 'make install' in the bin directory)
#install( TARGETS ${FINDING_CLUMPS_EXE} DESTINATION bin )
