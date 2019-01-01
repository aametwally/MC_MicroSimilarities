//
// Created by asem on 15/09/18.
//

#ifndef MARKOVIAN_FEATURES_MICROSIMILARITYVOTINGCLASSIFIER_HPP
#define MARKOVIAN_FEATURES_MICROSIMILARITYVOTINGCLASSIFIER_HPP

#include "MacroSimilarityClassifier.hpp"

namespace MC {
    template<size_t States>
    class MicroSimilarityVotingClassifier : public MacroSimilarityClassifier<States>
    {
        using Base = MacroSimilarityClassifier<States>;
        using BackboneProfiles = typename Base::BackboneProfiles;
        using ModelTrainer  = typename Base::ModelTrainer;
        using Similarity = typename Base::Similarity;
        using MCF = typename Base::MCF;
        using HeteroHistogramsFeatures  = typename Base::HeteroHistogramsFeatures;
    public:
        explicit MicroSimilarityVotingClassifier(
                const BackboneProfiles &backbones,
                const BackboneProfiles &background,
                const ModelTrainer modelTrainer,
                const Similarity similarity,
                std::optional<std::reference_wrapper<const Selection>> selection = std::nullopt )
                : Base( backbones, background, modelTrainer, similarity, selection )
        {
        }

        virtual ~MicroSimilarityVotingClassifier() = default;

    protected:
        ScoredLabels _predict( std::string_view sequence ) const override
        {
            std::map<std::string_view, double> voter;
            const size_t k = this->_backbones->get().size();

            if ( auto query = this->_modelTrainer( sequence, this->_selectedHistograms ); *query )
            {
                for (const auto &[order, isoHistograms] : query->histograms().get())
                {
                    for (const auto &[id, histogram1] : isoHistograms)
                    {
                        ScoredLabels pq( k );
                        for (const auto &[clusterName, profile] : this->_backbones->get())
                        {
                            auto &bg = this->_background->get().at( clusterName );
                            auto histogram2 = profile->histogram( order, id );
                            auto hBG = bg->histogram( order, id );
                            if ( histogram2 && hBG )
                            {
                                auto val = this->_similarity( histogram1, histogram2->get()) -
                                           this->_similarity( histogram1, hBG->get());
//                                    auto val = _similarity( histogram1 - hBG->get(), histogram2->get() - hBG->get());
                                pq.emplace( clusterName, val );
                            }
                        }

                        pq.forTopK( 5, [&]( const auto &candidate, size_t index ) {
                            std::string_view label = candidate.getLabel();
                            double val = (1) / (index + 1);
                            voter[label] += val;
                        } );
                    }
                }
            }

            voter = minmaxNormalize( std::move( voter ));

            ScoredLabels scoredQueue( k );
            for (auto[label, votes] : voter)
                scoredQueue.emplace( label, votes );
            for (auto &[label, profile] : this->_backbones->get())
            {
                scoredQueue.findOrInsert( label );
            }
            return scoredQueue;
        }
    };
}


#endif //MARKOVIAN_FEATURES_MICROSIMILARITYVOTINGCLASSIFIER_HPP
