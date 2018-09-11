//
// Created by asem on 19/08/18.
//

#ifndef MARKOVIAN_FEATURES_FEATURESCORINGPIPELINE_HPP
#define MARKOVIAN_FEATURES_FEATURESCORINGPIPELINE_HPP

#include "similarities.hpp"
#include "LabeledEntry.hpp"
#include "ConfusionMatrix.hpp"
#include "CrossValidationStatistics.hpp"
#include "crossvalidation.hpp"
#include "FeatureScoreAUC.hpp"
#include "VariantGenerator.hpp"
#include "RangedOrderMC.hpp"
#include "MCFeatures.hpp"
#include "Pipeline.hpp"

namespace MC {
    template<typename Grouping>
    class FeatureScoringPipeline
    {
    public:

    private:
        using Ops = MCOps<Grouping>;
        using MCModel = AbstractMC<Grouping>;
        using Histogram = typename MCModel::Histogram;
        using MCF = MCFeatures<Grouping>;
        using HeteroHistograms = typename MCModel::HeteroHistograms;
        using HeteroHistogramsFeatures = typename MCModel::HeteroHistogramsFeatures;
        using BackboneProfiles = typename MCModel::BackboneProfiles;
        using ModelTrainer = typename MCModel::ModelTrainer;
        using HistogramsTrainer = typename MCModel::HistogramsTrainer;

        using SimilarityFunction = std::function<double( const Histogram &, const Histogram & )>;

        using MicroHistogramsPrediction =
        std::unordered_map<Order, std::unordered_map<HistogramID, std::pair<std::string_view, double> >>;


        struct Prediction
        {
            explicit Prediction( Selection &&_histograms,
                                 std::string &&pred, double similarityScore ) :
                    histograms( std::move( _histograms )), predicted( std::move( pred )), score( similarityScore )
            {}

            const Selection histograms;
            const double score;
            const std::string predicted;
        };


        struct AUCRecorder
        {
            static AUCRecorder &getAUCRecorder()
            {
                static AUCRecorder recorder;
                return recorder;
            }

            void record( const std::string &scoringEqn,
                         const std::string &similaritylabel,
                         double score, bool tp )
            {
                _record[scoringEqn][similaritylabel].record( score, tp );
            }

            double auc( const std::string &scoringEqn,
                        const std::string &similarityLabel )
            {
                return _record.at( scoringEqn ).at( similarityLabel ).auc();
            }

            template<size_t indentation = 0>
            void report()
            {
                fmt::print( "{:<{}}General Statistics:\n", "", indentation );

                auto printRow = _printRowFunction<indentation + 4>();
                for (auto &[eqn, fAUC] : _record)
                {
                    fmt::print( "{:<{}}Scoring Label:[[{}]]:\n", "", indentation, eqn );
                    for (auto &[label, _fAUC] : fAUC)
                    {
                        const auto n = _fAUC.n();
                        double tp = double( _fAUC.tp()) / n;
                        double fn = double( _fAUC.fn()) / n;

                        printRow( fmt::format( "AUC ({})(n={})(tp={:.2f})(fn={:.2f})(range={})(nans={})", label,
                                               n, tp, fn, _fAUC.range(), _fAUC.nans()), _fAUC.auc());
                    }
                    fmt::print( "\n" );
                }
            }

            template<size_t indentation = 0>
            void reportToFile( const std::string &prefix )
            {
                namespace fs = std::experimental::filesystem;
                fs::create_directory( prefix.c_str());

                for (auto &[eqn, fAUC] : _record)
                {
                    for (auto &[sim, _fAUC] : fAUC)
                    {
                        double auc = _fAUC.auc();
                        const auto n = _fAUC.n();
                        double tp = double( _fAUC.tp()) / n;
                        double fn = double( _fAUC.fn()) / n;

                        std::ofstream reportFile;
                        const std::string fname = fmt::format( "[eqn:{}][sim:{}][auc:{}]"
                                                               "[n={}][tp={:.2f}][fn={:.2f}]"
                                                               "[range={}][nans={}]", eqn, sim,
                                                               auc, n, tp, fn, _fAUC.range(), _fAUC.nans());
                        const std::string filePath = io::join( {prefix, fname}, "/" );
                        fmt::print( "Writing to file:{}\n", filePath );
                        reportFile.open( filePath, std::ios::out );
                        reportFile << _fAUC.tpfn2String() << "\n";
                        reportFile << _fAUC.scores2String();
                        reportFile << "\n";
                        reportFile.close();
                    }
                }
            }

        private:
            AUCRecorder() = default;

            template<size_t indentation, size_t col1Width = 70>
            static auto _printRowFunction()
            {
                return [=]( const std::string col1, double col2 ) {
                    constexpr const char *fmtSpec = "{:<{}}{:<{}}:{:.4f}\n";
                    fmt::print( fmtSpec, "", indentation, col1, col1Width, col2 );
                };
            }

            template<size_t indentation, size_t col1Width = 70>
            static auto _printRowFunction( std::ofstream &s )
            {
                return [&]( const std::string col1, double col2 ) {
                    constexpr const char *fmtSpec = "{:<{}}{:<{}}:{:.4f}\n";
                    s << fmt::format( fmtSpec, "", indentation, col1, col1Width, col2 );
                };
            }

        private:
            std::map<std::string, std::map<std::string, FeatureScoreAUC>> _record;
        };

        struct PredictionFeatureCorrelator
        {

            static AUCRecorder &getAUCRecorder()
            {
                return AUCRecorder::getAUCRecorder();
            }

            static PredictionFeatureCorrelator &getFeatureCorrelator()
            {
                static PredictionFeatureCorrelator correlator;
                return correlator;
            }

            std::optional<std::string> assert_feature( Order order, HistogramID id )
            {
                for (const auto &[scoreLabel, features] : _featuresScores)
                {
                    auto &isoFeatures = features.at( order );
                    if ( isoFeatures.find( id ) == isoFeatures.cend())
                    {
                        auto ids = keys( isoFeatures );
                        std::sort( ids.begin(), ids.end());
                        return fmt::format( "[{}] Id:{} no found in:{}\n", scoreLabel, id, io::join2string( ids, " " ));
                    }
                }
                return std::nullopt;
            }

            void report()
            {
                getAUCRecorder().report();
            }

            void reportToFile( const std::string &prefix )
            {
                getAUCRecorder().reportToFile( prefix );
            }

            void addFeatureScoring( std::string &&label, HeteroHistogramsFeatures &&scoring )
            {
                _featuresScores.insert_or_assign( std::move( label ), std::move( scoring ));
            }


            void record_ACCUMULATIVE( const std::string &similarity,
                                      const MicroHistogramsPrediction &predictions,
                                      const std::string &trueLabel )
            {
                if ( predictions.empty()) return;

                std::map<std::string_view, double> macroSimilarity;
                std::map<std::string_view, double> totalScore;

                for (auto &[order, isoHistogramsPrediction] : predictions)
                    for (auto &[id, histogramPrediction] :  isoHistogramsPrediction)
                    {
                        std::string_view predicted = histogramPrediction.first;
                        double microSimilarity = histogramPrediction.second;
                        macroSimilarity[predicted] += microSimilarity;
                        for (auto &[scoringLabel, features] : _featuresScores)
                        {
                            totalScore[scoringLabel] += features.at( order ).at( id );
                        }
                    }

                std::pair<std::string_view, size_t> predicted = *macroSimilarity.begin();
                for (auto &[l, macroScore] : macroSimilarity)
                {
                    if ( macroScore > predicted.second )
                    {
                        predicted.first = l;
                        predicted.second = macroScore;
                    }
                }

                bool tp = trueLabel == predicted.first;
                for (auto &[scoringLabel, features] : _featuresScores)
                {
                    getAUCRecorder().record( scoringLabel, similarity, totalScore.at( scoringLabel ), tp );
                }
            }

            void record_MICRO( const std::string &similarity,
                               const MicroHistogramsPrediction &microPredictions,
                               const std::string &trueLabel )
            {
                if ( microPredictions.empty()) return;

                for (auto &[order, isoHistogramsPrediction] : microPredictions)
                    for (auto &[id, histogramPrediction] :  isoHistogramsPrediction)
                    {
                        std::string_view predicted = histogramPrediction.first;
                        double microSimilarity = histogramPrediction.second;
                        bool tp = trueLabel == histogramPrediction.first;

                        for (auto &[scoringLabel, features] : _featuresScores)
                        {
                            double score = features.at( order ).at( id );
                            getAUCRecorder().record( fmt::format( "micro_{}", scoringLabel ),
                                                     similarity, score, tp );
                        }
                    }
            }


            void record_VOTING( const std::string &similarity,
                                const MicroHistogramsPrediction &predictions,
                                const std::string &trueLabel )
            {
                if ( predictions.empty()) return;

                std::map<std::string_view, size_t> voter;
                std::map<std::string_view, double> totalScore;

                for (auto &[order, isoHistogramsPrediction] : predictions)
                    for (auto &[id, histogramPrediction] :  isoHistogramsPrediction)
                    {
                        std::string_view predicted = histogramPrediction.first;
                        voter[predicted] += 1;
                        for (auto &[scoringLabel, features] : _featuresScores)
                        {
                            totalScore[scoringLabel] += features.at( order ).at( id );
                        }
                    }

                std::pair<std::string_view, size_t> predicted = *voter.begin();
                for (auto &[l, votes] : voter)
                {
                    if ( votes > predicted.second )
                    {
                        predicted.first = l;
                        predicted.second = votes;
                    }
                }

                bool tp = trueLabel == predicted.first;
                for (auto &[scoringLabel, features] : _featuresScores)
                {
                    getAUCRecorder().record( scoringLabel, similarity, totalScore.at( scoringLabel ), tp );
                }
            }

        private:
            PredictionFeatureCorrelator() = default;

            std::map<std::string, HeteroHistogramsFeatures> _featuresScores;
        };

    public:

        FeatureScoringPipeline( ModelTrainer &&modelTrainer, HistogramsTrainer &&histogramsTrainer )
                : _modelTrainer( modelTrainer ), _histogramsTrainer( histogramsTrainer )
        {

        }

        static std::vector<LabeledEntry>
        reducedAlphabetEntries( std::vector<LabeledEntry> &&entries )
        {
            return LabeledEntry::reducedAlphabetEntries<Grouping>( entries );
        }

        static const std::map<std::string, SimilarityFunction> &getSimilarityFunctions()
        {
            static std::random_device rd;  //Will be used to obtain a seed for the random number engine
            static std::mt19937 gen( rd()); //Standard mersenne_twister_engine seeded with rd()
            static std::uniform_real_distribution<> dis( 0.0, 1.0 );


            static std::map<std::string, SimilarityFunction> m{
                    {Cosine::label,
                            []( const Histogram &k1, const Histogram &k2 ) { return Cosine::measure( k1, k2 ); }},
                    {KullbackLeiblerDivergence::label,
                            []( const Histogram &k1, const Histogram &k2 ) {
                                return KullbackLeiblerDivergence::measure( k1, k2 );
                            }},
                    {ChiSquared::label,
                            []( const Histogram &k1, const Histogram &k2 ) { return ChiSquared::measure( k1, k2 ); }},
                    {Intersection::label,
                            []( const Histogram &k1, const Histogram &k2 ) { return Intersection::measure( k1, k2 ); }},
                    {Gaussian::label,
                            []( const Histogram &k1, const Histogram &k2 ) { return Gaussian::measure( k1, k2 ); }},
                    {DensityPowerDivergence1::label,
                            []( const Histogram &k1, const Histogram &k2 ) {
                                return DensityPowerDivergence1::measure( k1, k2 );
                            }},
                    {DensityPowerDivergence2::label,
                            []( const Histogram &k1, const Histogram &k2 ) {
                                return DensityPowerDivergence2::measure( k1, k2 );
                            }},
                    {DensityPowerDivergence3::label,
                            []( const Histogram &k1, const Histogram &k2 ) {
                                return DensityPowerDivergence3::measure( k1, k2 );
                            }},
                    {ItakuraSaitu::label,
                            []( const Histogram &k1, const Histogram &k2 ) { return ItakuraSaitu::measure( k1, k2 ); }},
                    {"dummy",
                            [&]( const Histogram &k1, const Histogram &k2 ) {
                                return dis( gen );
                            }}
            };

            return m;
        }

        void classify_VALIDATION_SAMPLER( const std::vector<std::string> &queries,
                                          const std::vector<std::string> &trueLabels,
                                          BackboneProfiles &&targets,
                                          std::map<std::string, std::vector<std::string >> &&trainingSequences,
                                          Selection &&selection )
        {
//        for (auto i = 0; i < 5; ++i)
//        {
//            Selection newSelection;
//            for (auto &[order, ids] : selection)
//            {
//                std::set<HistogramID> newIds;
//                std::sample( ids.cbegin(), ids.cend(),
//                             std::inserter( newIds, std::begin( newIds )), size_t( 0.4 * ids.size()),
//                             std::mt19937{std::random_device{}()} );
//                newSelection.emplace( order, std::move( newIds ));
//            }
//            classify_VALIDATION( queries, trueLabels, targets, trainingSequences, std::move( newSelection ));
//        }
            classify_VALIDATION( queries, trueLabels, targets, trainingSequences, std::move( selection ));

        }

        void registerKernelsScores( const BackboneProfiles &targets,
                                    const std::map<std::string, std::vector<std::string >> &trainingSequences,
                                    const Selection &selection )
        {
            auto &correlator = PredictionFeatureCorrelator::getFeatureCorrelator();
            auto trainedHistograms = Ops::trainIndividuals( trainingSequences, _histogramsTrainer, selection );
            auto[withinClassRadius, populationRadius] = MCF::populationRadius( trainedHistograms, selection );

            correlator.addFeatureScoring( "ALL2WITHIN_UNIFORM",
                                          MCF::minMaxScale(
                                                  MCF::histogramRelevance_ALL2WITHIN_UNIFORM( withinClassRadius,
                                                                                              populationRadius )));

            correlator.addFeatureScoring( "ALL2MIN_UNIFORM",
                                          MCF::minMaxScale(
                                                  MCF::histogramRelevance_ALL2MIN_UNIFORM( withinClassRadius,
                                                                                           populationRadius )));

            correlator.addFeatureScoring( "MAX2MIN_UNIFORM",
                                          MCF::minMaxScale(
                                                  MCF::histogramRelevance_MAX2MIN_UNIFORM( withinClassRadius,
                                                                                           populationRadius )));


            correlator.addFeatureScoring( "informationRadius_UNIFORM",
                                          MCF::minMaxScale(  std::move( populationRadius )));
        }


        static std::pair<std::string_view, double>
        nearestProfile( const HeteroHistograms &query,
                        const BackboneProfiles &targets,
                        const SimilarityFunction &similarityFunction )
        {

            std::optional<std::pair<std::string_view, double> > prediction;
            for (const auto &[clusterName, profile] : targets)
            {
                double totalSimilarity = 0;
                for (const auto &[order, isoKernels1] : query)
                {
                    for (auto &[id, k1] : isoKernels1.get())
                    {
                        if ( auto histogram2Opt = profile.histogram( order, id ); histogram2Opt )
                        {
                            double similarityScore = similarityFunction( k1, histogram2Opt.value().get());
                            totalSimilarity += similarityScore;
                        }
                    }
                }
                if ( prediction )
                {
                    if ( totalSimilarity > prediction.value().second )
                        prediction = {clusterName, totalSimilarity};
                } else prediction = {clusterName, totalSimilarity};
            }
            return prediction.value();
        }

        static MicroHistogramsPrediction
        nearestHistograms( const HeteroHistograms &query,
                           const BackboneProfiles &targets,
                           const SimilarityFunction &similarityFunction )
        {
            MicroHistogramsPrediction histogramsPrediction;

            for (const auto &[order, isoKernels1] : query)
            {
                auto &_histogramsPrediction = histogramsPrediction[order];
                for (auto &[id, queryHistogram] : isoKernels1)
                {
                    auto &__histogramsPrediction = _histogramsPrediction[id];
                    std::optional<std::pair<std::string_view, double>> closestKernel;
                    for (const auto &[clusterName, profile] : targets)
                    {
                        if ( auto histogram2Opt = profile->histogram( order, id ); histogram2Opt )
                        {
                            double similarityScore = similarityFunction( queryHistogram, histogram2Opt.value().get());
                            if ( closestKernel )
                            {
                                if ( similarityScore > closestKernel->second )
                                {
                                    closestKernel = {clusterName, similarityScore};
                                }
                            } else closestKernel = {clusterName, similarityScore};
                        }
                    }
                    if ( closestKernel )
                        _histogramsPrediction[id] = std::make_pair( closestKernel->first, closestKernel->second );
                }
            }

            return histogramsPrediction;
        }

        void classify_VALIDATION( const std::vector<std::string> &queries,
                                  const std::vector<std::string> &trueLabels,
                                  const BackboneProfiles &targets,
                                  const std::map<std::string, std::vector<std::string >> &trainingSequences,
                                  Selection &&selection )
        {
            assert( queries.size() == trueLabels.size());
            auto &correlator = PredictionFeatureCorrelator::getFeatureCorrelator();
            registerKernelsScores( targets, trainingSequences, selection );
            for (auto[similarityLabel, similarity] : getSimilarityFunctions())
            {
                for (auto queryIdx = 0; queryIdx < queries.size(); ++queryIdx)
                {
                    const auto &q = queries.at( queryIdx );
                    const auto &trueLabel = trueLabels.at( queryIdx );

                    if ( auto queryOpt = _histogramsTrainer( {q}, selection ); queryOpt )
                    {
                        auto &query = queryOpt.value();
                        auto histogramsPrediction = nearestHistograms( query, targets, similarity );
                        correlator.record_VOTING( similarityLabel, histogramsPrediction, trueLabel );
                        correlator.record_ACCUMULATIVE( similarityLabel, histogramsPrediction, trueLabel );
                        correlator.record_MICRO( similarityLabel, histogramsPrediction, trueLabel );
                    }
                }
            }
        }


        void runPipeline_VALIDATION( std::vector<LabeledEntry> &&entries,
                                     size_t k,
                                     const std::string &outputFile )
        {
            std::set<std::string> labels;
            for (const auto &entry : entries)
                labels.insert( entry.getLabel());

            using Folds = std::vector<std::vector<std::pair<std::string, std::string >>>;

            const Folds folds = [&]() {
                fmt::print( "[All Sequences:{}]\n", entries.size());
                entries = reducedAlphabetEntries( std::move( entries ));
                auto groupedEntries = LabeledEntry::groupSequencesByLabels( std::move( entries ));

                auto _labels = keys( groupedEntries );
                for (auto &l : _labels) l = fmt::format( "{}({})", l, groupedEntries.at( l ).size());

                fmt::print( "[Clusters:{}][{}]\n",
                            groupedEntries.size(),
                            io::join( _labels, "|" ));

                return kFoldStratifiedSplit( std::move( groupedEntries ), k );
            }();


            auto extractTest = []( const std::vector<std::pair<std::string, std::string >> &items ) {
                std::vector<std::string> sequences, labels;
                for (const auto &item : items)
                {
                    labels.push_back( item.first );
                    sequences.push_back( item.second );
                }
                return std::make_pair( sequences, labels );
            };

            CrossValidationStatistics validation( k, labels );
            std::unordered_map<long, size_t> histogram;

            for (auto i = 0; i < k; ++i)
            {
                auto trainingClusters = joinFoldsExceptK( folds, i );
                auto[test, tLabels] = extractTest( folds.at( i ));
                auto selection = Ops::withinJointAllUnionKernels( trainingClusters, _histogramsTrainer, 0.05 );
                auto trainedProfiles = Ops::train( trainingClusters, _modelTrainer, selection );
                classify_VALIDATION_SAMPLER( test, tLabels, std::move( trainedProfiles ),
                                             std::move( trainingClusters ), std::move( selection ));
            }

            PredictionFeatureCorrelator::getFeatureCorrelator().report();
            PredictionFeatureCorrelator::getFeatureCorrelator().reportToFile( outputFile );
        }


    private:
        const ModelTrainer _modelTrainer;
        const HistogramsTrainer _histogramsTrainer;
    };


    using FeatureScoringPipelineVariant = MakeVariantType<FeatureScoringPipeline, SupportedAAGrouping>;

    template<typename AAGrouping>
    FeatureScoringPipelineVariant getFeatureScoringPipeline( const std::string &model,
                                                             Order mnOrder, Order mxOrder )
    {
        const MCModelsEnum modelEnum = MCModelLabels.at( model );
        using RMC = MC<AAGrouping>;
        using ROMC = RangedOrderMC<AAGrouping>;
        using ZMC = ZYMC<AAGrouping>;
        auto modelLabel = MCModelLabels.at( model );
        switch (modelLabel)
        {
            case MCModelsEnum::RegularMC :
                return FeatureScoringPipeline<AAGrouping>( RMC::getModelTrainer( mxOrder ),
                                                           RMC::getHistogramsTrainer( mxOrder ));
            case MCModelsEnum::RangedOrderMC :
                return FeatureScoringPipeline<AAGrouping>( ROMC::getModelTrainer( mnOrder, mxOrder ),
                                                           ROMC::getHistogramsTrainer( mnOrder, mxOrder ));
            case MCModelsEnum::ZhengYuanMC :
                return FeatureScoringPipeline<AAGrouping>( ZMC::getModelTrainer( mxOrder ),
                                                           ZMC::getHistogramsTrainer( mxOrder ));
            default:
                throw std::runtime_error( "Undefined Strategy" );
        }
    };

    FeatureScoringPipelineVariant getFeatureScoringPipeline( const std::string &groupingLabel,
                                                             const std::string &model,
                                                             Order mnOrder, Order mxOrder )
    {
        const AminoAcidGroupingEnum grouping = GroupingLabels.at( groupingLabel );
        switch (grouping)
        {
            case AminoAcidGroupingEnum::NoGrouping20:
                return getFeatureScoringPipeline<AAGrouping_NOGROUPING20>( model, mnOrder, mxOrder );
            case AminoAcidGroupingEnum::DIAMOND11 :
                return getFeatureScoringPipeline<AAGrouping_DIAMOND11>( model, mnOrder, mxOrder );
            case AminoAcidGroupingEnum::OFER8 :
                return getFeatureScoringPipeline<AAGrouping_OFER8>( model, mnOrder, mxOrder );
            case AminoAcidGroupingEnum::OFER15 :
                return getFeatureScoringPipeline<AAGrouping_OFER15>( model, mnOrder, mxOrder );
        }
    };
}
#endif //MARKOVIAN_FEATURES_FEATURESCORINGPIPELINE_HPP
