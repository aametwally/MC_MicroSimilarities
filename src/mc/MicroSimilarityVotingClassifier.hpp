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
                : Base( backbones, background, selection, modelTrainer, similarity )
        {
        }

        virtual std::vector<std::string_view> predict( const std::vector<std::string> &test ) const override
        {
            if ( this->_backbones && this->_background )
            {
                auto clustersIR = MCF::informationRadius_UNIFORM( this->_backbones->get(), this->_selectedHistograms );
                auto backgroundIR = MCF::informationRadius_UNIFORM( this->_background->get(),
                                                                    this->_selectedHistograms );

                std::vector<std::string_view> labels;
                for (auto &seq : test)
                    labels.emplace_back( _predict( seq, clustersIR, backgroundIR ));
                return labels;
            } else throw std::runtime_error( fmt::format( "Bad training" ));
        }

    protected:
        std::string_view _predict( std::string_view sequence,
                                   const HeteroHistogramsFeatures &clustersIR,
                                   const HeteroHistogramsFeatures &backgroundIR ) const
        {
            using PriorityQueue = typename MatchSet<Score>::Queue<std::string_view>;


            PriorityQueue matchSet( this->_backbones->get().size());

            std::map<std::string_view, double> voter;

            if ( auto query = this->_modelTrainer( sequence, this->_selectedHistograms ); *query )
            {
                for (const auto &[order, isoHistograms] : query->histograms().get())
                {
                    for (const auto &[id, histogram1] : isoHistograms)
                    {
                        PriorityQueue pq( this->_backbones->get().size());
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
                        double score = getOr( backgroundIR, order, id, double( 0 )) -
                                       getOr( clustersIR, order, id, double( 0 ));

                        pq.forTopK( 5, [&]( const auto &candidate, size_t index ) {
                            std::string_view label = candidate.getLabel();
                            voter[label] += (1 + score) / (index + 1);
                        } );
                    }
                }
            }

            auto top = std::pair< std::string_view, double>( unclassified, -inf );
            for (auto[label, votes] : voter)
            {
                if ( votes > top.second )
                    top = {label, votes};
            }
            return top.first;
        }
    };


}


#endif //MARKOVIAN_FEATURES_MICROSIMILARITYVOTINGCLASSIFIER_HPP
