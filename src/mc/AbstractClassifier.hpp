//
// Created by asem on 18/09/18.
//

#ifndef MARKOVIAN_FEATURES_ABSTRACTCLASSIFIER_HPP
#define MARKOVIAN_FEATURES_ABSTRACTCLASSIFIER_HPP

#include "SimilarityMetrics.hpp"
#include "MCDefs.h"

namespace MC {

    enum class ClassificationEnum
    {
        Voting,
        Accumulative,
        Propensity,
        DiscriminativePropensity,
        Segmentation,
        SVM,
        KNN,
        KMERS,
        SVM_Stack,
        KNN_Stack
    };

    static const std::map<std::string, ClassificationEnum> ClassifierEnum = {
            {"voting",         ClassificationEnum::Voting},
            {"acc",            ClassificationEnum::Accumulative},
            {"propensity",     ClassificationEnum::Propensity},
            {"discriminative", ClassificationEnum::DiscriminativePropensity},
            {"segmentation",   ClassificationEnum::Segmentation},
            {"svm",            ClassificationEnum::SVM},
            {"knn",            ClassificationEnum::KNN},
            {"svm_stack",      ClassificationEnum::SVM_Stack},
            {"knn_stack",      ClassificationEnum::KNN_Stack},
            {"kmers",          ClassificationEnum::KMERS}
    };

    static const std::map<ClassificationEnum, std::string_view> ClassifierLabel = [&]() {
        std::map<ClassificationEnum, std::string_view> m;
        for (auto &[label, enumm] : ClassifierEnum)
            m.emplace( enumm, label );
        return m;
    }();

    class AbstractClassifier
    {

    public:
        virtual ~AbstractClassifier()
        {}

    public:
        std::vector<std::string_view> predict( const std::vector<std::string> &sequences ) const
        {
            assert( _validTraining());
            std::vector<std::string_view> labels;
            for (auto &seq : sequences)
                labels.emplace_back( _bestPrediction( seq ));

            return labels;
        }

        std::vector<ScoredLabels> scoredPredictions( const std::vector<std::string> &sequences ) const
        {
            assert( _validTraining());
            std::vector<ScoredLabels> scoredLabels;
            for (auto &seq : sequences)
                scoredLabels.emplace_back( scoredPredictions( seq ));

            return scoredLabels;
        }

        ScoredLabels scoredPredictions( std::string_view sequence ) const
        {
            return _predict( sequence );
        }

    protected:
        virtual bool _validTraining() const = 0;

        virtual std::string_view _bestPrediction( std::string_view sequence ) const
        {
            auto predictions = _predict( sequence );
            if ( auto top = predictions.top(); top )
            {
                return top->get().getLabel();
            } else return unclassified;
        }

        virtual ScoredLabels _predict( std::string_view sequence ) const = 0;

    protected:

    };

}
#endif //MARKOVIAN_FEATURES_ABSTRACTCLASSIFIER_HPP
