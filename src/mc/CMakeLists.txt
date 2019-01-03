find_package( dlib )
set( MC_LIB mc_lib )

set( MC_SOURCES
        Selection.cpp
        AbstractMC.cpp
        AbstractClassifier.cpp
        MCKmersClassifier.cpp
        MCPropensityClassifier.cpp
        MCDiscriminativeClassifier.cpp
        MCSegmentationClassifier.cpp
        MacroSimilarityClassifier.cpp
        MicroSimilarityVotingClassifier.cpp
        MCDiscretizedScalesClassifier.cpp
        KNNConfusionMC.cpp
        KNNMCParameters.cpp
        KNNModel.cpp
        GappedMC.cpp
        PolymorphicMC.cpp
        MC.cpp
        Pipeline.cpp
        SVMConfusionMC.cpp
        SVMMCParameters.cpp
        SVMModel.cpp
        PolymorphicZYMC.cpp
        ZYMC.cpp
        DiscretizedScalesMC.cpp )


set( MC_DETAILS_TEMPLATE_FILES
        AbstractMC.hpp
        MCKmersClassifier.hpp
        AbstractClassifier.hpp
        MCPropensityClassifier.hpp
        MCDiscriminativeClassifier.hpp
        MCSegmentationClassifier.hpp
        MCDiscretizedScalesClassifier.hpp
        MacroSimilarityClassifier.hpp
        MicroSimilarityVotingClassifier.hpp
        KNNConfusionMC.hpp
        KNNMCParameters.hpp
        KNNModel.hpp
        GappedMC.hpp
        PolymorphicMC.hpp
        MC.hpp
        Pipeline.hpp
        SVMConfusionMC.hpp
        SVMMCParameters.hpp
        SVMModel.hpp
        PolymorphicZYMC.hpp
        ZYMC.hpp
        MLConfusedMC.hpp
        DiscretizedScalesMC.hpp )

set( MC_INCLUDES
        Pipeline.hpp
        Series.hpp
        VariantGenerator.hpp
        FeatureScoringPipeline.hpp
        Histogram.hpp
        MCDefs.h
        Selection.hpp )



add_library( ${MC_LIB} STATIC ${MC_SOURCES} ${MC_DETAILS_TEMPLATE_FILES} ${MC_INCLUDES}  )
add_dependencies(${MC_LIB}  dlib fmt )

target_include_directories( ${MC_LIB} PUBLIC
        ${EXTERNAL_INSTALL_LOCATION}/include
        ${CMAKE_SOURCE_DIR}/src/include
        ../report
        ../features
        ../data
        ../segmentation
        )

target_link_libraries( ${MC_LIB} stdc++fs dlib::dlib
        ${SEGMENTATION_LIB}
        ${REPORT_LIB}
        ${FEATURES_LIB}
        ${DATA_LIB} )

target_compile_features( ${MC_LIB} PUBLIC cxx_std_17)

set( MC_INCLUDES ${MC_INCLUDES} PARENT_SCOPE )
set( MC_LIB ${MC_LIB} PARENT_SCOPE )