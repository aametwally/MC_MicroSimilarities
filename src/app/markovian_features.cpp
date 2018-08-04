#include "markovian_features.hpp"
#include "clara.hpp"


int main(int argc, char *argv[])
{
    std::string input, testFile;
    int markovianOrder = 2;
    int gappedMarkovianOrder = -1;
    float testPercentage = 0.1f, threshold = -1;
    bool showHelp = false;
    auto cli
            = clara::Arg(input, "input")
                      ("UniRef input file")
              | clara::Opt(testFile, "test")
              ["-T"]["--test"]
                      ("test file")
              | clara::Opt(markovianOrder, "order")
              ["-o"]["--order"]
                      ("Markovian order")
              | clara::Opt(gappedMarkovianOrder, "gorder")
              ["-g"]["--gorder"]
                      ("Gapped markovian order")
              | clara::Opt(testPercentage, "percentage")
              ["-p"]["--percentage"]
                      ("test percentage")
              | clara::Opt(threshold, "threshold")
              ["-t"]["--t"]
                      ("Below cluster size average to exclude from subsetting")
              | clara::Help(showHelp);

    auto result = cli.parse(clara::Args(argc, argv));
    if (!result) {
        fmt::print("Error in command line:{}\n", result.errorMessage());
        exit(1);
    } else if (showHelp) {
        cli.writeToStream(std::cout);
    } else if (!testFile.empty()) {
        fmt::print("[Args][input:{}][testFile:{}][order:{}][testPercentage:{}][threshold:{}]\n",
                   input, testFile, markovianOrder, testPercentage, threshold);


    } else {
        fmt::print("[Args][input:{}][order:{}][testPercentage:{}][threshold:{}]\n",
                   input, markovianOrder, testPercentage, threshold);

        PipelineVariant pipeline = getConfiguredPipeline("diamond11", "chi");


        std::vector<UniRefEntry> entries = UniRefEntry::fasta2UnirefEntries(FastaEntry::readFastaFile(input));
        input.clear();

        std::visit([&](auto &&p) {
            p.runPipeline_VALIDATION(std::move(entries),
                                     markovianOrder,
                                     testPercentage,
                                     threshold);
        }, pipeline);
    }
    return 0;
}
