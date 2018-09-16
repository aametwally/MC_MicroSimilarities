//
// Created by asem on 15/09/18.
//

#ifndef MARKOVIAN_FEATURES_MACROSIMILARITYCLASSIFIER_HPP
#define MARKOVIAN_FEATURES_MACROSIMILARITYCLASSIFIER_HPP

#include "AbstractMC.hpp"
#include "MCFeatures.hpp"

#include "SimilarityMetrics.hpp"

namespace MC {
    template<typename Grouping>
    class MacroSimilarityClassifier
    {
    protected:
        using MCModel = AbstractMC<Grouping>;
        using Histogram = typename MCModel::Histogram;
        using Similarity = MetricFunction<Histogram>;
        using MCF = MCFeatures<Grouping>;
        using HeteroHistograms = typename MCModel::HeteroHistograms;
        using HeteroHistogramsFeatures = typename MCModel::HeteroHistogramsFeatures;
        using BackboneProfiles = typename MCModel::BackboneProfiles;
        using ModelTrainer =  ModelGenerator<Grouping>;

    public:

        explicit MacroSimilarityClassifier( const BackboneProfiles &backbones,
                                            const BackboneProfiles &background,
                                            const Selection &selection,
                                            const ModelTrainer modelTrainer,
                                            const Similarity similarityFunction )
                : _backbones( backbones ), _background( background ),
                  _modelTrainer( modelTrainer ), _selectedHistograms( selection ),
                  _similarity( similarityFunction )
        {

        }


        virtual std::vector<std::string_view> predict( const std::vector<std::string> &test ) const
        {
            if ( _backbones && _background )
            {
                std::vector<std::string_view> labels;
                for (auto &seq : test)
                    labels.emplace_back( _predict( seq ));
                return labels;
            } else throw std::runtime_error( fmt::format( "Bad training" ));
        }


    protected:

        std::string_view _predict( std::string_view sequence ) const
        {
            using PriorityQueue = typename MatchSet<Score>::Queue<std::string_view>;

            PriorityQueue matchSet( _backbones->get().size());

            if ( auto query = _modelTrainer( sequence, _selectedHistograms ); *query )
            {
                for (const auto &[clusterId, profile] : _backbones->get())
                {
                    auto &bg = _background->get().at( clusterId );
                    double sum = 0;
                    for (const auto &[order, isoKernels] : query->histograms().get())
                        for (const auto &[id, histogram1] : isoKernels)
                        {
//                                double score = getOr( relevance, order, id, double( 0 ));
                            auto histogram2 = profile->histogram( order, id );
                            auto hBG = bg->histogram( order, id );
                            if ( histogram2 && hBG )
                            {
                                sum += _similarity( histogram1, histogram2->get()) -
                                       _similarity( histogram1, hBG->get());
//                                    sum += similarity( histogram1 - hBG->get(), histogram2->get() - hBG->get());
                            }
                        }
                    matchSet.emplace( clusterId, sum );
                }
            }

            if ( auto top = matchSet.top(); top )
                return top->get().getLabel();
            else return unclassified;
        }

    protected:
        std::optional<std::reference_wrapper<const BackboneProfiles >> _backbones;
        std::optional<std::reference_wrapper<const BackboneProfiles >> _background;
        const Selection &_selectedHistograms;
        const ModelTrainer _modelTrainer;
        const Similarity _similarity;
    };

}
#endif //MARKOVIAN_FEATURES_MACROSIMILARITYCLASSIFIER_HPP
