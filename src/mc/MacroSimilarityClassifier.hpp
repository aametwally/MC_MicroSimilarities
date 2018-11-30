//
// Created by asem on 15/09/18.
//

#ifndef MARKOVIAN_FEATURES_MACROSIMILARITYCLASSIFIER_HPP
#define MARKOVIAN_FEATURES_MACROSIMILARITYCLASSIFIER_HPP

#include "AbstractMC.hpp"
#include "MCFeatures.hpp"

#include "SimilarityMetrics.hpp"
#include "AbstractClassifier.hpp"

namespace MC {
    template<typename Grouping>
    class MacroSimilarityClassifier : public AbstractClassifier
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
                                            const ModelTrainer modelTrainer,
                                            const Similarity similarityFunction ,
                                            std::optional<std::reference_wrapper<const Selection>> selection = std::nullopt )
                : _backbones( backbones ), _background( background ),
                  _modelTrainer( modelTrainer ), _selectedHistograms( selection ),
                  _similarity( similarityFunction )
        {

        }

        virtual ~MacroSimilarityClassifier() = default;

    protected:
        bool _validTraining() const override
        {
            return _backbones && _background
                   && _backbones->get().size() == _background->get().size();
        }

        ScoredLabels _predict( std::string_view sequence ) const override
        {
            std::map<std::string_view, double> macros;
            if ( auto query = _modelTrainer( sequence, _selectedHistograms ); *query )
            {
                for (const auto &[label, profile] : _backbones->get())
                {
                    auto &bg = _background->get().at( label );
                    double macro = 0;
                    for (const auto &[order, isoKernels] : query->histograms().get())
                        for (const auto &[id, histogram1] : isoKernels)
                        {
//                                double _score = getOr( relevance, order, id, double( 0 ));
                            auto histogram2 = profile->histogram( order, id );
                            auto hBG = bg->histogram( order, id );
                            if ( histogram2 && hBG )
                            {
                                macro += _similarity( histogram1, histogram2->get()) -
                                         _similarity( histogram1, hBG->get());
//                                    sum += similarity( histogram1 - hBG->get(), histogram2->get() - hBG->get());
                            }
                        }

                    macros[label] = macro;
                }
            }

            macros = minmaxNormalize( std::move( macros ));

            ScoredLabels matchSet( _backbones->get().size());
            for (auto &[label, macro] : macros)
                matchSet.emplace( label, macro );

            for( auto &[label,profile] : this->_backbones->get() )
            {
                matchSet.findOrInsert( label );
            }

            return matchSet;
        }

    protected:
        std::optional<std::reference_wrapper<const BackboneProfiles >> _backbones;
        std::optional<std::reference_wrapper<const BackboneProfiles >> _background;
        std::optional<std::reference_wrapper<const Selection>> _selectedHistograms;
        const ModelTrainer _modelTrainer;
        const Similarity _similarity;
    };

}
#endif //MARKOVIAN_FEATURES_MACROSIMILARITYCLASSIFIER_HPP
