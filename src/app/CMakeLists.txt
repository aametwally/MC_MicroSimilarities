## Dependencies

set( COMMON_INCLUDE_DIRECTORIES
        ../mc
        ../report
        ../data
        ../features )

set( MC_SOURCE_FILES
        ../mc/Selection.cpp
        ../mc/AbstractMC.cpp
        ../mc/MacroSimilarityClassifier.cpp
        ../mc/MicroSimilarityVotingClassifier.cpp
        ../mc/KNNConfusionMC.cpp
        ../mc/KNNMCParameters.cpp
        ../mc/KNNModel.cpp
        ../mc/LSMC.cpp
        ../mc/MC.cpp
        ../mc/Pipeline.cpp
        ../mc/SVMConfusionMC.cpp
        ../mc/SVMMCParameters.cpp
        ../mc/SVMModel.cpp
        ../mc/ZYMC.cpp ../report/ConfusionMatrix.cpp ../report/CrossValidationStatistics.cpp)


set( MC_DETAILS_TEMPLATE_FILES
        ../mc/AbstractMC.hpp
        ../mc/MacroSimilarityClassifier.hpp
        ../mc/MicroSimilarityVotingClassifier.hpp
        ../mc/KNNConfusionMC.hpp
        ../mc/KNNMCParameters.hpp
        ../mc/KNNModel.hpp
        ../mc/LSMC.hpp
        ../mc/MC.hpp
        ../mc/Pipeline.hpp
        ../mc/SVMConfusionMC.hpp
        ../mc/SVMMCParameters.hpp
        ../mc/SVMModel.hpp
        ../mc/ZYMC.hpp
        ../mc/MLConfusedMC.hpp)

set( COMMON_INCLUDES
        ../mc/common.hpp
        ../report/Timers.hpp
        ../data/SequenceEntry.hpp
        ../data/FastaEntry.hpp
        ../data/LabeledEntry.hpp
        ../report/ConfusionMatrix.hpp
        ../mc/SimilarityMetrics.hpp
        ../mc/aminoacids_grouping.hpp
        ../report/CrossValidationStatistics.hpp
        ../report/crossvalidation.hpp
        ../mc/Pipeline.hpp
        ../mc/Series.hpp
        ../mc/VariantGenerator.hpp
        ../data/dlib_utilities.hpp
        ../features/FeatureScoreAUC.hpp
        ../mc/FeatureScoringPipeline.hpp
        ../mc/Histogram.hpp
        ../mc/MCDefs.h
        ../mc/Selection.hpp)

add_definitions(-ftemplate-backtrace-limit=0)

find_package( dlib )
#message( ${dlib_LIBS} )
### Generate the MC Pipeline executable
add_executable( MC_pipeline ${COMMON_INCLUDES} ${MC_SOURCE_FILES} ${MC_DETAILS_TEMPLATE_FILES} MC_pipeline.cpp)
target_include_directories( MC_pipeline PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${COMMON_INCLUDE_DIRECTORIES}
        ${dlib_INCLUDE_DIRS}  )

target_link_libraries( MC_pipeline PRIVATE fmt-header-only stdc++fs dlib::dlib )
target_compile_features( MC_pipeline PRIVATE cxx_std_17)

### Generate the feature scoring executable
add_executable( markovian_features_scoring  ${COMMON_INCLUDES}  ${MC_SOURCE_FILES}
        ${MC_DETAILS_TEMPLATE_FILES} markovian_features_analysis.cpp ../mc/AbstractMC.cpp)
target_include_directories( markovian_features_scoring PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${COMMON_INCLUDE_DIRECTORIES}
        ${dlib_INCLUDE_DIRS} )
target_link_libraries( markovian_features_scoring PRIVATE fmt-header-only stdc++fs  dlib::dlib )
target_compile_features( markovian_features_scoring PRIVATE cxx_std_17 )

### Confusion Matrix Test
add_executable( confusionMatrix  ${COMMON_INCLUDES} confusionMatrix.cpp ../mc/MCFeatures.cpp ../mc/RangedOrderMC.cpp)
target_include_directories( confusionMatrix PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${COMMON_INCLUDE_DIRECTORIES} )
target_link_libraries( confusionMatrix PRIVATE fmt-header-only  )
target_compile_features( confusionMatrix PRIVATE cxx_std_17)


# Install the executable (generate with 'make install' in the bin directory)
#install( TARGETS ${FINDING_CLUMPS_EXE} DESTINATION bin )
