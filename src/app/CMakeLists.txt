## Dependencies

set( COMMON_INCLUDE_DIRECTORIES
        ../algorithm
        ../report
        ../data
        ../features )

set( COMMON_INCLUDES
        ../algorithm/common.hpp
        ../algorithm/HOMC.hpp
        ../report/Timers.hpp
        ../data/SequenceEntry.hpp
        ../data/FastaEntry.hpp
        ../data/LabeledEntry.hpp
        ../report/ConfusionMatrix.hpp
        ../features/similarities.hpp
        ../algorithm/aminoacids_grouping.hpp
        ../report/CrossValidationStatistics.hpp
        ../report/crossvalidation.hpp
        ../algorithm/HOMCPipeline.hpp
        ../algorithm/Series.hpp
        ../algorithm/SVMPipeline.hpp
        ../algorithm/VariantGenerator.hpp
        ../algorithm/SVMMarkovianModel.hpp
        ../data/dlib_utilities.hpp
        ../features/FeatureScoreAUC.hpp
        ../algorithm/FeatureScoringPipeline.hpp
        ../algorithm/HOMCFeatures.hpp
        ../algorithm/Histogram.hpp
        ../algorithm/HOMCOperations.hpp
        ../algorithm/HOMCDefs.hpp
        ../algorithm/MarkovChains.hpp
        ../algorithm/ZYMC.hpp
        ../algorithm/ZYPipeline.hpp)

add_definitions(-ftemplate-backtrace-limit=0)
### Generate the MC Pipeline executable
add_executable( homc_pipeline ${COMMON_INCLUDES} HOMC_pipeline.cpp)
target_include_directories( homc_pipeline PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${COMMON_INCLUDE_DIRECTORIES} )
target_link_libraries( homc_pipeline PRIVATE fmt-header-only stdc++fs )
target_compile_features( homc_pipeline PRIVATE cxx_std_17)

### Generate the Zheng Yuan Pipeline executable
add_executable( ZhengYuan_pipeline ${COMMON_INCLUDES} ZhengYuan_pipeline.cpp)
target_include_directories( ZhengYuan_pipeline PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${COMMON_INCLUDE_DIRECTORIES} )
target_link_libraries( ZhengYuan_pipeline PRIVATE fmt-header-only stdc++fs )
target_compile_features( ZhengYuan_pipeline PRIVATE cxx_std_17)


### Generate the main executable
#find_library (DLIB_LIBRARY NAMES libdlib dlilb libdlib.a PATHS ${DLIB_PREFIX}/lib NO_DEFAULT_PATH)

add_executable( markovian_features_svm ${COMMON_INCLUDES} markovian_features_svm.cpp)
target_include_directories( markovian_features_svm PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${COMMON_INCLUDE_DIRECTORIES}
        ${DLIB_PREFIX}/include )

target_link_libraries( markovian_features_svm PRIVATE fmt-header-only stdc++fs dlib::dlib )
target_compile_features( markovian_features_svm PRIVATE cxx_std_17)

### Generate the subsetter executable
add_executable( subsetDataset  ${COMMON_INCLUDES} subsetDataset.cpp)
target_include_directories( subsetDataset PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${COMMON_INCLUDE_DIRECTORIES} )
target_link_libraries( subsetDataset PRIVATE fmt-header-only  stdc++fs )
target_compile_features( subsetDataset PRIVATE cxx_std_17)

### Generate the feature scoring executable
add_executable( markovian_features_scoring  ${COMMON_INCLUDES} markovian_features_analysis.cpp)
target_include_directories( markovian_features_scoring PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${COMMON_INCLUDE_DIRECTORIES} )
target_link_libraries( markovian_features_scoring PRIVATE fmt-header-only stdc++fs )
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
