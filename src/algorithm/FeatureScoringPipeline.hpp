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
#include "MarkovianKernels.hpp"
#include "MarkovianModelFeatures.hpp"

template<typename Grouping>
class FeatureScoringPipeline
{
    using MF = MarkovianModelFeatures<Grouping>;
    using MP = MarkovianKernels<Grouping>;
public:

private:
    using MarkovianProfile = MarkovianKernels<Grouping>;
    using Kernel = typename MarkovianProfile::Kernel;
    using MarkovianProfiles = std::map<std::string, MarkovianProfile>;
    using KernelID = typename MarkovianProfile::KernelID;
    using Order = typename MarkovianProfile::Order;

    using HeteroKernels =  typename MarkovianProfile::HeteroKernels;
    using HeteroKernelsFeatures =  typename MarkovianProfile::HeteroKernelsFeatures;

    using DoubleSeries = typename MarkovianProfile::ProbabilitisByOrder;
    using KernelsSeries = typename MarkovianProfile::KernelSeriesByOrder;
    using KernelsSelection = std::unordered_map<Order, std::set<KernelID >>;

    static constexpr Order MinOrder = MarkovianProfile::MinOrder;
    static constexpr size_t StatesN = MarkovianProfile::StatesN;
    static constexpr double eps = std::numeric_limits<double>::epsilon();
    static constexpr double inf = std::numeric_limits<double>::infinity();

    using Selection = typename MarkovianProfile::Selection;
    using SimilarityFunction = std::function<double( const Kernel &, const Kernel & )>;
    using KernelIdentifier = typename MarkovianProfile::KernelIdentifier;

    struct KernelPrediction
    {
        explicit KernelPrediction( KernelIdentifier &&kid, std::string &&pred, double similarityScore ) :
                kernel( kid ), predicted( std::move( pred )), similarity( similarityScore )
        {}

        const double similarity;
        const KernelIdentifier kernel;
        const std::string predicted;
    };

    struct Prediction
    {
        explicit Prediction( Selection &&_kernels,
                             std::string &&pred, double similarityScore ) :
                kernels( std::move( _kernels )), predicted( std::move( pred )), score( similarityScore )
        {}

        const Selection kernels;
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
            fs::create_directory( prefix.c_str() );

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
                    const std::string filePath = io::join( {prefix, fname} , "/");
                    fmt::print("Writing to file:{}\n", filePath);
                    reportFile.open( filePath , std::ios::out );
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

        std::optional<std::string> assert_feature( Order order, KernelID id )
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

        void addFeatureScoring( std::string &&label, HeteroKernelsFeatures &&scoring )
        {
            _featuresScores.insert_or_assign( std::move( label ), std::move( scoring ));
        }


        void record_ACCUMULATIVE( const std::string &similarity,
                                  const Prediction &&prediction,
                                  const std::string &trueLabel )
        {

            std::map<std::string, std::map<std::string, double>> totalDifferentialScore;
            std::map<std::string, double> totalDifferentialOrder;
            for (const auto &[order, ids] : prediction.kernels)
            {
                for (auto id : ids)
                {
                    auto predicted = prediction.predicted;
                    totalDifferentialOrder[predicted] += order;
                    for (auto &[scoringLabel, features] : _featuresScores)
                    {
                        totalDifferentialScore[scoringLabel][predicted] += features.at( order ).at( id );
                    }
                }

            }

            for (const auto &[scoringLabel, predicted] : totalDifferentialScore)
            {
                const std::string scoreLabel = fmt::format( "differential_{}", scoringLabel );
                for (const auto&[_predicted, dscore] :predicted)
                {
                    getAUCRecorder().record( scoreLabel, similarity, dscore, _predicted == trueLabel );
                }
            }
            const std::string scoreLabel = "differential_order";
            for (const auto&[predicted, dscore] :totalDifferentialOrder)
            {
                getAUCRecorder().record( scoreLabel, similarity, dscore, predicted == trueLabel );
            }
        }

        void record_MICRO( const std::string &similarity,
                           const std::vector<KernelPrediction> &predictions,
                           const std::string &trueLabel )
        {
            if ( predictions.empty()) return;

            std::map<std::string, double> totalScore;

            for (const KernelPrediction &p : predictions)
            {
                Order order = p.kernel.order;
                KernelID id = p.kernel.id;
                bool tp = trueLabel == p.predicted;

                for (auto &[scoringLabel, features] : _featuresScores)
                {
                    getAUCRecorder().record( fmt::format( "micro_{}", scoringLabel ),
                                             similarity, features.at( order ).at( id ), tp );
                }
                getAUCRecorder().record( "micro_order", similarity, order, tp );
            }

        }


        void record_VOTING( const std::string &similarity,
                            const std::vector<KernelPrediction> &predictions,
                            const std::string &trueLabel )
        {
            if ( predictions.empty()) return;

            std::map<std::string, size_t> voter;
            std::map<std::string, double> totalScore;
            double totalOrders = 0;

            for (const KernelPrediction &p : predictions)
            {
                Order order = p.kernel.order;
                KernelID id = p.kernel.id;
                auto predicted = p.predicted;
                totalOrders += order;
                voter[p.predicted] += p.similarity;
                for (auto &[scoringLabel, features] : _featuresScores)
                {
                    totalScore[scoringLabel] += features.at( order ).at( id );
                }
            }

            std::pair<std::string, size_t> predicted = *voter.begin();
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
            getAUCRecorder().record( "order", similarity, totalOrders, tp );
        }

    private:
        PredictionFeatureCorrelator() = default;

        std::map<std::string, HeteroKernelsFeatures> _featuresScores;
    };

public:

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
                        []( const Kernel &k1, const Kernel &k2 ) { return Cosine::measure( k1, k2 ); }},
                {KullbackLeiblerDivergence::label,
                        []( const Kernel &k1, const Kernel &k2 ) {
                            return KullbackLeiblerDivergence::measure( k1, k2 );
                        }},
                {ChiSquared::label,
                        []( const Kernel &k1, const Kernel &k2 ) { return ChiSquared::measure( k1, k2 ); }},
                {Intersection::label,
                        []( const Kernel &k1, const Kernel &k2 ) { return Intersection::measure( k1, k2 ); }},
                {Gaussian::label,
                        []( const Kernel &k1, const Kernel &k2 ) { return Gaussian::measure( k1, k2 ); }},
                {DensityPowerDivergence1::label,
                        []( const Kernel &k1, const Kernel &k2 ) {
                            return DensityPowerDivergence1::measure( k1, k2 );
                        }},
                {DensityPowerDivergence2::label,
                        []( const Kernel &k1, const Kernel &k2 ) {
                            return DensityPowerDivergence2::measure( k1, k2 );
                        }},
                {DensityPowerDivergence3::label,
                        []( const Kernel &k1, const Kernel &k2 ) {
                            return DensityPowerDivergence3::measure( k1, k2 );
                        }},
                {ItakuraSaitu::label,
                        []( const Kernel &k1, const Kernel &k2 ) { return ItakuraSaitu::measure( k1, k2 ); }},
                {"dummy",
                        [&]( const Kernel &k1, const Kernel &k2 ) {
                            return dis( gen );
                        }}
        };

        return m;
    }

    static void classify_VALIDATION_SAMPLER( const std::vector<std::string> &queries,
                                             const std::vector<std::string> &trueLabels,
                                             MarkovianProfiles &&targets,
                                             std::map<std::string, std::vector<std::string >> &&trainingSequences,
                                             Selection &&selection )
    {
//        for (auto i = 0; i < 5; ++i)
//        {
//            Selection newSelection;
//            for (auto &[order, ids] : selection)
//            {
//                std::set<KernelID> newIds;
//                std::sample( ids.cbegin(), ids.cend(),
//                             std::inserter( newIds, std::begin( newIds )), size_t( 0.4 * ids.size()),
//                             std::mt19937{std::random_device{}()} );
//                newSelection.emplace( order, std::move( newIds ));
//            }
//            classify_VALIDATION( queries, trueLabels, targets, trainingSequences, std::move( newSelection ));
//        }
        classify_VALIDATION( queries, trueLabels, targets, trainingSequences, std::move( selection ));

    }

    static void registerKernelsScores( Order order,
                                       const MarkovianProfiles &targets,
                                       const std::map<std::string, std::vector<std::string >> &trainingSequences,
                                       const Selection &selection )
    {
        auto &correlator = PredictionFeatureCorrelator::getFeatureCorrelator();
        correlator.addFeatureScoring( "ALL2WITHIN_WEIGHTED",
                                      MF::minMaxScale(
                                              MF::histogramRelevance_ALL2WITHIN_WEIGHTED( trainingSequences, order,
                                                                                          selection )));
        correlator.addFeatureScoring( "ALL2WITHIN_UNIFORM",
                                      MF::minMaxScale(
                                              MF::histogramRelevance_ALL2WITHIN_UNIFORM( trainingSequences, order,
                                                                                         selection )));
        correlator.addFeatureScoring( "ALL2MIN_WEIGHTED",
                                      MF::minMaxScale(
                                              MF::histogramRelevance_ALL2MIN_WEIGHTED( trainingSequences, order,
                                                                                       selection )));
        correlator.addFeatureScoring( "ALL2MIN_UNIFORM",
                                      MF::minMaxScale( MF::histogramRelevance_ALL2MIN_UNIFORM( trainingSequences, order,
                                                                                               selection )));
        correlator.addFeatureScoring( "MAX2MIN_WEIGHTED",
                                      MF::minMaxScale(
                                              MF::histogramRelevance_MAX2MIN_WEIGHTED( trainingSequences, order,
                                                                                       selection )));
        correlator.addFeatureScoring( "MAX2MIN_UNIFORM",
                                      MF::minMaxScale( MF::histogramRelevance_MAX2MIN_UNIFORM( trainingSequences, order,
                                                                                               selection )));

        correlator.addFeatureScoring( "informationRadius_WEIGHTED",
                                      MF::minMaxScale( MF::informationRadius_WEIGHTED( targets, MF::histogramWeights(
                                              targets ))));
        correlator.addFeatureScoring( "informationRadius_UNIFORM",
                                      MF::minMaxScale( MF::informationRadius_UNIFORM( targets )));
    }


    static std::pair<std::string, double>
    nearestProfile( const MarkovianProfile &query,
                    const MarkovianProfiles &targets,
                    const SimilarityFunction &similarityFunction )
    {

        std::optional<std::pair<std::string, double> > prediction;
        for (const auto &[clusterName, profile] : targets)
        {
            double totalSimilarity = 0;
            for (const auto &[order, isoKernels1] : query.kernels())
            {
                for (auto &[id, k1] : isoKernels1.get())
                {

                    const double p1 = double( k1.hits()) / query.totalCharacters();

                    if ( auto kernel2Opt = profile.kernel( order, id ); kernel2Opt )
                    {
                        double similarityScore = similarityFunction( k1, kernel2Opt.value().get());
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

    static std::vector<KernelPrediction>
    nearestKernels( const MarkovianProfile &query,
                    const MarkovianProfiles &targets,
                    const SimilarityFunction &similarityFunction )
    {
        std::vector<KernelPrediction> kernelPredictions;

        for (const auto &[order, isoKernels1] : query.kernels())
        {
            for (auto &[id, k1] : isoKernels1.get())
            {
                std::optional<std::pair<std::string, double>> closestKernel;
                for (const auto &[clusterName, profile] : targets)
                {
                    if ( auto kernel2Opt = profile.kernel( order, id ); kernel2Opt )
                    {
                        double similarityScore = similarityFunction( k1, kernel2Opt.value().get());
                        if ( closestKernel )
                        {
                            if ( similarityScore > closestKernel.value().second )
                            {
                                closestKernel = {clusterName, similarityScore};
                            }
                        } else closestKernel = {clusterName, similarityScore};
                    }
                }
                if ( closestKernel )
                    kernelPredictions.push_back( KernelPrediction( KernelIdentifier( order, id ),
                                                                   std::move( closestKernel.value().first ),
                                                                   closestKernel.value().second ));
            }
        }

        return kernelPredictions;
    }

    static void classify_VALIDATION( const std::vector<std::string> &queries,
                                     const std::vector<std::string> &trueLabels,
                                     const MarkovianProfiles &targets,
                                     const std::map<std::string, std::vector<std::string >> &trainingSequences,
                                     Selection &&selection )
    {
        assert( queries.size() == trueLabels.size());
        const Order mxOrder = MarkovianProfile::maxOrder( targets );
        auto &correlator = PredictionFeatureCorrelator::getFeatureCorrelator();
        registerKernelsScores( mxOrder, targets, trainingSequences, selection );
        for (auto[similarityLabel, similarity] : getSimilarityFunctions())
        {
            for (auto queryIdx = 0; queryIdx < queries.size(); ++queryIdx)
            {
                const auto &q = queries.at( queryIdx );
                const auto &trueLabel = trueLabels.at( queryIdx );

                if ( auto queryOpt = MarkovianProfile::filter( MarkovianProfile( {q}, mxOrder ), selection ); queryOpt )
                {
                    auto &query = queryOpt.value();
                    std::vector<KernelPrediction> kernelPredictions = nearestKernels( query, targets, similarity );
                    correlator.record_VOTING( similarityLabel, kernelPredictions, trueLabel );
                    correlator.record_MICRO( similarityLabel, kernelPredictions, trueLabel );

                    auto predictedProfile = nearestProfile( query, targets, similarity );
                    Prediction p( query.featureSpace(), std::move( predictedProfile.first ), predictedProfile.second );
                    correlator.record_ACCUMULATIVE( similarityLabel, std::move( p ), trueLabel );
                }
            }
        }
    }


    void runPipeline_VALIDATION( std::vector<LabeledEntry> &&entries,
                                 Order order,
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
            auto selection = MF::withinJointAllUnionKernels( trainingClusters, order, 0.05 );
            auto trainedProfiles = MP::train( trainingClusters , order , selection );
            classify_VALIDATION_SAMPLER( test, tLabels, std::move( trainedProfiles ),
                                         std::move( trainingClusters ), std::move( selection ));
        }

        PredictionFeatureCorrelator::getFeatureCorrelator().report();
        PredictionFeatureCorrelator::getFeatureCorrelator().reportToFile( outputFile );
    }

};


using FeatureScoringPipelineVariant = MakeVariantType<FeatureScoringPipeline, SupportedAAGrouping>;

FeatureScoringPipelineVariant getFeatureScoringPipeline( const std::string &groupingLabel )
{
    const AminoAcidGroupingEnum grouping = GroupingLabels.at( groupingLabel );
    switch (grouping)
    {
        case AminoAcidGroupingEnum::NoGrouping20:
            return FeatureScoringPipeline<AAGrouping_NOGROUPING20>();
        case AminoAcidGroupingEnum::DIAMOND11 :
            return FeatureScoringPipeline<AAGrouping_DIAMOND11>();
        case AminoAcidGroupingEnum::OLFER8 :
            return FeatureScoringPipeline<AAGrouping_OLFER8>();
        case AminoAcidGroupingEnum::OLFER15 :
            return FeatureScoringPipeline<AAGrouping_OLFER15>();
    }
};

#endif //MARKOVIAN_FEATURES_FEATURESCORINGPIPELINE_HPP
