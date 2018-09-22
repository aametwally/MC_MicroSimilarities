//
// Created by asem on 15/09/18.
//

#ifndef MARKOVIAN_FEATURES_MICROSIMILARITYVOTINGCLASSIFIER_HPP
#define MARKOVIAN_FEATURES_MICROSIMILARITYVOTINGCLASSIFIER_HPP

#include "MacroSimilarityClassifier.hpp"

namespace MC {
    template<typename Grouping>
    class MicroSimilarityVotingClassifier : public MacroSimilarityClassifier<Grouping>
    {
        using Base = MacroSimilarityClassifier<Grouping>;
        using BackboneProfiles = typename Base::BackboneProfiles;
        using ModelTrainer  = typename Base::ModelTrainer;
        using Similarity = typename Base::Similarity;
        using MCF = typename Base::MCF;
        using HeteroHistogramsFeatures  = typename Base::HeteroHistogramsFeatures;
    public:
        explicit MicroSimilarityVotingClassifier( const BackboneProfiles &backbones,
                                                  const BackboneProfiles &background,
                                                  const Selection &selection,
                                                  const ModelTrainer modelTrainer,
                                                  const Similarity similarity )
                : Base( backbones, background, selection, modelTrainer, similarity ),
                  _clustersIR( MCF::informationRadius_UNIFORM( backbones , selection )),
                  _backgroundIR( MCF::informationRadius_UNIFORM( background , selection ))
        {
        }

    protected:
        ScoredLabels _predict( std::string_view sequence  ) const override
        {
            std::map<std::string_view, double> voter;

            if ( auto query = this->_modelTrainer( sequence, this->_selectedHistograms ); *query )
            {
                for (const auto &[order, isoHistograms] : query->histograms().get())
                {
                    for (const auto &[id, histogram1] : isoHistograms)
                    {
                        ScoredLabels pq( this->_backbones->get().size());
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
                        double score = getOr( _backgroundIR, order, id, double( 0 )) -
                                       getOr( _clustersIR, order, id, double( 0 ));

                        pq.forTopK( 5, [&]( const auto &candidate, size_t index ) {
                            std::string_view label = candidate.getLabel();
                            double val = (1 + score) / (index + 1);
                            voter[label] += val;
                        } );
                    }
                }
            }

            voter = minmaxNormalize( std::move( voter ));

            ScoredLabels scoredQueue( this->_backbones->get().size());
            for (auto[label, votes] : voter)
                scoredQueue.emplace( label, votes  );

            return scoredQueue;
        }

    protected:
        const HeteroHistogramsFeatures _clustersIR;
        const HeteroHistogramsFeatures _backgroundIR;
    };
}


#endif //MARKOVIAN_FEATURES_MICROSIMILARITYVOTINGCLASSIFIER_HPP
