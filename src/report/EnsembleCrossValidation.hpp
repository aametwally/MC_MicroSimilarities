//
// Created by asem on 16/09/18.
//

#ifndef MARKOVIAN_FEATURES_ENSEMBLECROSSVALIDATION_HPP
#define MARKOVIAN_FEATURES_ENSEMBLECROSSVALIDATION_HPP

#include "SimilarityMetrics.hpp"

#include "CrossValidationStatistics.hpp"
#include "LabeledEntry.hpp"
#include "FeatureScoreAUC.hpp"

template<typename Label = std::string_view>
class EnsembleCrossValidation
{
public:
    using ItemID = std::string_view;
    using ClassLabel = Label;
    using PredictionLabel = Label;
    using FoldID = size_t;
    using ClassifierLabel = std::string_view;
    using Fold = std::vector<std::pair<std::string, LabeledEntry >>;
    using Folds = std::vector<Fold>;

    static constexpr std::string_view unclassified = std::string_view();


    explicit EnsembleCrossValidation( const Folds &folds )
            : _k( folds.size()),
              _actualLabels( _getActualLabels( folds )),
              _features( _getFeatures( folds ))
    {}

    void countInstance(
            FoldID k,
            ClassifierLabel classifier,
            ItemID id,
            ScoredLabels predictions
    )
    {
        assert( k >= 0 && k < _k );
        assert( _actualLabels.find( id ) != _actualLabels.cend());
        _predictions[k][classifier].emplace_back( id, std::move( predictions ));
    }

    void countInstance(
            FoldID k,
            ClassifierLabel classifier,
            ItemID id,
            ScoredLabels predictions,
            Label label
    )
    {
        assert( _actualLabels.at( id ) == label );
        assert( k >= 0 && k < _k );
        assert( _actualLabels.find( id ) != _actualLabels.cend());
        _predictions[k][classifier].emplace_back( id, std::move( predictions ));
    }


    std::vector<std::tuple<
            std::vector<ClassifierLabel>,
            CrossValidationStatistics<Label>,
            std::map<std::string_view, FeatureScoreAUC >>>
    majorityVotingOverallAccuracy() const
    {
        return ensembleOverallAccuracy( [this](
                auto &combination,
                auto &assignment
        ) {
            return _majorityVoting( combination, assignment );
        } );
    }

    std::vector<std::tuple<
            std::vector<ClassifierLabel>,
            CrossValidationStatistics<Label>,
            std::map<std::string_view, FeatureScoreAUC >>>
    weightedVotingOverallAccuracy() const
    {
        return ensembleOverallAccuracy( [this](
                auto &combination,
                auto &assignment
        ) {
            return _weightedVoting( combination, assignment );
        } );
    }

    template<typename VotingMethod>
    std::vector<std::tuple<
            std::vector<ClassifierLabel>,
            CrossValidationStatistics<Label>,
            std::map<std::string_view, FeatureScoreAUC >>>
    ensembleOverallAccuracy( VotingMethod votingMethod ) const
    {
        std::vector<std::tuple<
                std::vector<ClassifierLabel>,
                CrossValidationStatistics<Label>,
                std::map<std::string_view, FeatureScoreAUC> >> ensembleCrossValidation;

        for (auto &combination : _getClassifiersCombinations())
        {
            CrossValidationStatistics<Label> validation( _k, _getLabels());
            std::map<std::string_view, FeatureScoreAUC> auc;
            for (auto &[k, assignments] : _predictions)
            {
                auto predictions = votingMethod( combination, assignments );
                for (auto &[id, predicted] : predictions)
                {
                    std::string_view actualLabel = _actualLabels.at( id );
                    validation.countInstance( k, predicted, actualLabel );
                    for (auto &[feature, value] : _features.at( id ))
                        auc[feature].record( value, predicted == actualLabel );
                }
            }
            ensembleCrossValidation.emplace_back( combination, validation, auc );
        }
        return ensembleCrossValidation;
    }

private:
    static std::map<ItemID, ClassLabel>
    _getActualLabels(
            const std::vector<ItemID> &members,
            const std::vector<Label> &labels
    )
    {
        assert( members.size() == labels.size());
        std::map<ItemID, ClassLabel> m;
        for (size_t i = 0; i < members.size(); ++i)
            m[members.at( i )] = labels.at( i );
        return m;
    }

    static std::map<ItemID, ClassLabel>
    _getActualLabels( const Folds &folds )
    {
        std::map<ItemID, ClassLabel> m;
        std::vector<std::pair<ItemID, ClassLabel  >> v;
        for (auto &fold : folds)
            for (auto &[label, entry] : fold)
                v.emplace_back( entry.memberId(), label );


        for (auto &[id, label] : v)
            m.emplace( id, label );

        assert( v.size() == m.size());
        return m;
    }

    static std::map<ItemID, std::map<std::string, double>>
    _getFeatures( const Folds &folds )
    {
        std::map<ItemID, std::map<std::string, double>> m;
        for (auto &fold : folds)
            for (auto &[label, entry] : fold)
                m[entry.memberId()]["length"] = entry.length();

        return m;
    }


    std::vector<std::vector<ClassifierLabel >>
    _getClassifiersCombinations() const
    {
        assert( !_predictions.empty());
        std::set<ClassifierLabel> classifierSet;
        for (const auto &[classifier, predictions] : _predictions.begin()->second)
            classifierSet.insert( classifier );
        std::vector<ClassifierLabel> classifierVector( classifierSet.begin(), classifierSet.end());
        return combinations( classifierVector );
    }

    std::set<Label>
    _getLabels() const
    {
        std::set<Label> labels;
        for (const auto &[id, label] : _actualLabels)
            labels.insert( label );
        return labels;
    }


    std::vector<std::pair<ItemID, PredictionLabel>>
    _majorityVoting(
            const std::vector<ClassifierLabel> &voters,
            const std::map<ClassifierLabel, std::vector<std::pair<ItemID, ScoredLabels >>> &predictions
    ) const
    {
        std::map<ItemID, std::map<PredictionLabel, size_t >> votes;
        for (auto &voter : voters)
            for (auto&[id, prediction] : predictions.at( voter ))
                if ( auto top = prediction.top(); top )
                    ++votes[id][top->get().label()];

        std::vector<std::pair<ItemID, PredictionLabel>> majorityPredictions;
        for (auto &[id, _votes] : votes)
        {
            auto top = std::pair<PredictionLabel, size_t>( unclassified, 0 );
            for (auto &[prediction, __votes] : _votes)
            {
                if ( __votes > top.second )
                {
                    top = {prediction, __votes};
                }
            }
            majorityPredictions.emplace_back( id, top.first );
        }
        return majorityPredictions;
    }

    std::vector<std::pair<ItemID, PredictionLabel>>
    _weightedVoting(
            const std::vector<ClassifierLabel> &voters,
            const std::map<ClassifierLabel, std::vector<std::pair<ItemID, ScoredLabels >>> &predictions
    ) const
    {
        std::map<ItemID, std::map<PredictionLabel, size_t >> votes;
        for (auto &voter : voters)
            for (auto&[id, prediction] : predictions.at( voter ))
                if ( auto top = prediction.top(); top )
                    votes[id][top->get().label()] += top->get().value();

        std::vector<std::pair<ItemID, PredictionLabel>> majorityPredictions;
        for (auto &[id, _votes] : votes)
        {
            auto top = std::pair<PredictionLabel, size_t>( unclassified, 0 );
            for (auto &[prediction, __votes] : _votes)
            {
                if ( __votes > top.second )
                {
                    top = {prediction, __votes};
                }
            }
            majorityPredictions.emplace_back( id, top.first );
        }
        return majorityPredictions;
    }

private:

    const FoldID _k;
    const std::map<ItemID, ClassLabel> _actualLabels;
    const std::map<ItemID, std::map<std::string, double>> _features;
    std::unordered_map<FoldID,
            std::map<ClassifierLabel,
                    std::vector<std::pair<ItemID, ScoredLabels  >>>> _predictions;
};


#endif //MARKOVIAN_FEATURES_ENSEMBLECROSSVALIDATION_HPP
