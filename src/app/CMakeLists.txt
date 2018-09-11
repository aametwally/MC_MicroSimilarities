## Dependencies

set( COMMON_INCLUDE_DIRECTORIES
        ../algorithm
        ../report
        ../data
        ../features )

set( COMMON_INCLUDES
        ../algorithm/common.hpp
        ../algorithm/RangedOrderMC.hpp
        ../report/Timers.hpp
        ../data/SequenceEntry.hpp
        ../data/FastaEntry.hpp
        ../data/LabeledEntry.hpp
        ../report/ConfusionMatrix.hpp
        ../features/similarities.hpp
        ../algorithm/aminoacids_grouping.hpp
        ../report/CrossValidationStatistics.hpp
        ../report/crossvalidation.hpp
        ../algorithm/Pipeline.hpp
        ../algorithm/Series.hpp
        ../algorithm/VariantGenerator.hpp
        ../algorithm/SVMMarkovianModel.hpp
        ../data/dlib_utilities.hpp
        ../features/FeatureScoreAUC.hpp
        ../algorithm/FeatureScoringPipeline.hpp
        ../algorithm/MCFeatures.hpp
        ../algorithm/Histogram.hpp
        ../algorithm/MCOperations.hpp
        ../algorithm/AbstractMC.hpp
        ../algorithm/ZYMC.hpp
        ../algorithm/ZYPipeline.hpp ../algorithm/MC.hpp ../algorithm/MCPipeline.hpp)

add_definitions(-ftemplate-backtrace-limit=0)

find_package( dlib )
#message( ${dlib_LIBS} )
### Generate the MC Pipeline executable
add_executable( MC_pipeline ${COMMON_INCLUDES} MC_pipeline.cpp)
target_include_directories( MC_pipeline PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${COMMON_INCLUDE_DIRECTORIES}
        ${dlib_INCLUDE_DIRS}  )

target_link_libraries( MC_pipeline PRIVATE fmt-header-only stdc++fs dlib::dlib )
target_compile_features( MC_pipeline PRIVATE cxx_std_17)

### Generate the feature scoring executable
add_executable( markovian_features_scoring  ${COMMON_INCLUDES} markovian_features_analysis.cpp)
target_include_directories( markovian_features_scoring PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${COMMON_INCLUDE_DIRECTORIES}
        ${dlib_INCLUDE_DIRS} )
target_link_libraries( markovian_features_scoring PRIVATE fmt-header-only stdc++fs  dlib::dlib )
target_compile_features( markovian_features_scoring PRIVATE cxx_std_17 )

### Confusion Matrix Test
add_executable( confusionMatrix  ${COMMON_INCLUDES} confusionMatrix.cpp )
target_include_directories( confusionMatrix PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${COMMON_INCLUDE_DIRECTORIES} )
target_link_libraries( confusionMatrix PRIVATE fmt-header-only  )
target_compile_features( confusionMatrix PRIVATE cxx_std_17)


# Install the executable (generate with 'make install' in the bin directory)
#install( TARGETS ${FINDING_CLUMPS_EXE} DESTINATION bin )
