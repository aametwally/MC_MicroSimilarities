set( MC_LIB mc_lib )

set( MC_SOURCES
  Selection.cpp
  AbstractMCClassifier.cpp
  MCKmersClassifier.cpp
  MCSegmentationClassifier.cpp
  MCDiscretizedScalesClassifier.cpp
  KNNModel.cpp
  Pipeline.cpp
  SVMModel.cpp
  )


set( MC_DETAILS_TEMPLATE_FILES
  SparseTransitionMatrix.hpp
  AbstractMC.hpp
  MCKmersClassifier.hpp
  AbstractMCClassifier.hpp
  MCSegmentationClassifier.hpp
  MCDiscretizedScalesClassifier.hpp
  KNNModel.hpp
  Pipeline.hpp
  SVMModel.hpp
  MCBasedMLModel.hpp
  )

set( MC_INCLUDES
  MCModels.hpp
  Pipeline.hpp
  Series.hpp
  VariantGenerator.hpp
  FeatureScoringPipeline.hpp
  Histogram.hpp
  MCDefs.h
  Selection.hpp )


add_library( ${MC_LIB} STATIC ${MC_SOURCES} ${MC_DETAILS_TEMPLATE_FILES} ${MC_INCLUDES} )

target_include_directories( ${MC_LIB} PUBLIC
  ${EXTERNAL_INSTALL_LOCATION}/include
  ${CMAKE_SOURCE_DIR}/src/include
  ../report
  ../features
  ../data
  ../segmentation
  )

target_link_libraries( ${MC_LIB} stdc++fs CONAN_PKG::dlib CONAN_PKG::fmt
  ${SEGMENTATION_LIB}
  ${REPORT_LIB}
  ${FEATURES_LIB}
  ${DATA_LIB} )

target_compile_features( ${MC_LIB} PUBLIC cxx_std_17 )

set( MC_INCLUDES ${MC_INCLUDES} PARENT_SCOPE )
set( MC_LIB ${MC_LIB} PARENT_SCOPE )