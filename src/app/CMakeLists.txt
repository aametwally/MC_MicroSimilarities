## Dependencies
#include( Qt )
include_directories( ../algorithm )

### Generate the executable
add_executable( markovian_features   markovian_features.cpp)

# Install the executable (generate with 'make install' in the bin directory)
#install( TARGETS ${FINDING_CLUMPS_EXE} DESTINATION bin )
