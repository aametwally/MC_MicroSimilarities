## Dependencies


set( COMMON_INCLUDE_DIRECTORIES
        ../algorithm
        ../report
        ../data ../features )
set( COMMON_INCLUDES
        ../algorithm/common.hpp
        ../algorithm/MarkovianKernels.hpp
        ../report/Timers.hpp
        ../data/SequenceEntry.hpp
        ../data/FastaEntry.hpp
        ../data/LabeledEntry.hpp ../report/ConfusionMatrix.hpp ../features/similarities.hpp ../algorithm/aminoacids_grouping.hpp ../report/CrossValidationStatistics.hpp ../report/crossvalidation.hpp ../algorithm/ConfiguredPipeline.hpp ../algorithm/Series.hpp)

### Generate the main executable
add_executable( markovian_features ${COMMON_INCLUDES} markovian_features.cpp)
target_include_directories( markovian_features PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${COMMON_INCLUDE_DIRECTORIES} )
target_link_libraries( markovian_features PRIVATE fmt-header-only stdc++fs )
target_compile_features( markovian_features PRIVATE cxx_std_17)

### Generate the subsetter executable
add_executable( subsetDataset  ${COMMON_INCLUDES} subsetDataset.cpp)
target_include_directories( subsetDataset PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${COMMON_INCLUDE_DIRECTORIES} )
target_link_libraries( subsetDataset PRIVATE fmt-header-only  stdc++fs )
target_compile_features( subsetDataset PRIVATE cxx_std_17)


### Confusion Matrix Test
add_executable( confusionMatrix  ${COMMON_INCLUDES} confusionMatrix.cpp )
target_include_directories( confusionMatrix PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${COMMON_INCLUDE_DIRECTORIES} )
target_link_libraries( confusionMatrix PRIVATE fmt-header-only  )
target_compile_features( confusionMatrix PRIVATE cxx_std_17)


# Install the executable (generate with 'make install' in the bin directory)
#install( TARGETS ${FINDING_CLUMPS_EXE} DESTINATION bin )
