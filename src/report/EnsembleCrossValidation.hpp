//
// Created by asem on 16/09/18.
//

#ifndef MARKOVIAN_FEATURES_ENSEMBLECROSSVALIDATION_HPP
#define MARKOVIAN_FEATURES_ENSEMBLECROSSVALIDATION_HPP

#include "CrossValidationStatistics.hpp"
#include "LabeledEntry.hpp"

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

    EnsembleCrossValidation( FoldID k, const std::vector<ItemID> &members,
                             const std::vector<Label> &labels )
            : _k( k ), _actualLabels( _getActualLabels( members, labels ))
    {}

    explicit EnsembleCrossValidation( const Folds &folds )
            : _k( folds.size()), _actualLabels( _getActualLabels( folds ))
    {}

    void countInstance( FoldID k, ClassifierLabel classifier, ItemID id, PredictionLabel prediction )
    {
        assert( k >= 0 && k < _k );
        assert( _actualLabels.find( id ) != _actualLabels.cend());
        _predictions[k][classifier].emplace_back( id, prediction );
    }

    void countInstance( FoldID k, ClassifierLabel classifier, ItemID id, PredictionLabel prediction , Label label )
    {
        assert( _actualLabels.at( id ) == label );
        assert( k >= 0 && k < _k );
        assert( _actualLabels.find( id ) != _actualLabels.cend());
        _predictions[k][classifier].emplace_back( id, prediction );
    }


    std::vector<std::pair<std::vector<ClassifierLabel>, CrossValidationStatistics<Label> >>
    majorityVotingOverallAccuracy() const
    {
        std::vector<std::pair<std::vector<ClassifierLabel>, CrossValidationStatistics<Label> >> majorityCrossValidation;

        for (auto &combination : _getClassifiersCombinations())
        {
            CrossValidationStatistics<Label> validation( _k, _getLabels());
            for (auto &[k, assignments] : _predictions)
            {
                auto majorityPredictions = _majorityVoting( combination, assignments );
                for (auto &[id, predicted] : majorityPredictions)
                {

                    validation.countInstance( k, predicted, _actualLabels.at( id ));
                }
            }
            majorityCrossValidation.emplace_back( combination, validation );
        }
        return majorityCrossValidation;
    }

private:
    static std::map<ItemID, ClassLabel>
    _getActualLabels( const std::vector<ItemID> &members, const std::vector<Label> &labels )
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
                v.emplace_back( entry.getMemberId(), label );


        for (auto &[id, label] : v)
            m.emplace( id, label );

        assert( v.size() == m.size());
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
    _majorityVoting( const std::vector<ClassifierLabel> &voters,
                     const std::map<ClassifierLabel, std::vector<std::pair<ItemID, PredictionLabel>>> &predictions ) const
    {
        std::map<ItemID, std::map<PredictionLabel, size_t >> votes;
        for (auto &voter : voters)
            for (auto&[id, prediction] : predictions.at( voter ))
                ++votes[id][prediction];

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

    std::unordered_map<FoldID,
            std::map<ClassifierLabel,
                    std::vector<std::pair<ItemID, PredictionLabel >>>> _predictions;
};


#endif //MARKOVIAN_FEATURES_ENSEMBLECROSSVALIDATION_HPP
