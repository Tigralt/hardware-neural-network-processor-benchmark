#include <cnpy.h>      // https://github.com/rogersce/cnpy
#include <cxxopts.hpp> // https://github.com/jarro2783/cxxopts
#include <regex>
#include <chrono>
#include "dma.hpp"
#include "tqdm.hpp"

void system_pause()
{
    std::cout << "Press enter to continue ...";
    std::cin.get();
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("npu_tester", "Software to test NPU with different neural network architectures and datasets");
    options.add_options()
        ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
        ("c,core", "Number of core in the NPU (REQUIRED)", cxxopts::value<int>())
        ("d,dir", "Directory in which are layers.npz and datasets.npz files (REQUIRED)", cxxopts::value<std::string>())
        ("h,help", "Print usage")
    ;

    auto result = options.parse(argc, argv);

    if (result.count("help") || result.count("dir") == 0 || result.count("core") == 0)
    {
      std::cout << options.help() << std::endl;
      exit(0);
    }

    unsigned int verbosity_level = result.count("verbose");
    std::string dir = result["dir"].as<std::string>();
    size_t core = result["core"].as<int>();

    tqdm bar;
    unsigned long int status, mem_status;
    size_t correct_classification = 0, dst_length = 0, execution_time = 0;
    std::vector<float> results;
    std::string layers_file("layers.npz");
    std::string dataset_file("dataset.npz");

    cnpy::npz_t layers = cnpy::npz_load(dir + layers_file);
    cnpy::npz_t dataset = cnpy::npz_load(dir + dataset_file);
    float *input = dataset["x"].data<float>();
    char *output = dataset["y"].data<char>();

    mmap_params config_src = {0x30100000, 65536};
    mmap_params weight_src = {0x30110000, 33554432};
    mmap_params io_src = {0x32110000, 262144};
    mmap_params io_dst = {0x32130000, 262144};

    DirectMemoryAccess *config = new DirectMemoryAccess(0x40400000, &config_src, NULL);
    DirectMemoryAccess *weight = new DirectMemoryAccess(0x40410000, &weight_src, NULL);
    DirectMemoryAccess *io = new DirectMemoryAccess(0x40420000, &io_src, &io_dst);

    // Instructions number
    config->writeSourceUInt64(layers.size());

    // Load weights and instructions
    for (cnpy::npz_t::iterator it = layers.begin(); it != layers.end(); it++)
    {
        if (verbosity_level > 1)
        {
            std::cout << "Loading layer \"" << it->first << "..." << std::endl;
        }

        // Instructions
        std::regex re("a\\d+\\_([a-z]+)\\_\\d+");
        std::smatch match;
        std::regex_search(it->first, match, re);
        std::string result = match.str(1);
        unsigned int activation = 0;
        if (result == "sigmoid")
        {
            activation = 1;
        }
        else if (result == "relu")
        {
            activation = 2;
        }
        else if (result == "softmax")
        {
            activation = 3;
        }

        uint64_t layer_input_shape = it->second.shape[0];
        uint64_t layer_output_shape = it->second.shape[1];
        uint64_t activation_cast = activation;
        uint64_t instruction = (layer_input_shape << 34) + (layer_output_shape << 4) + activation_cast;
        
        config->writeSourceUInt64(instruction);
        dst_length = it->second.shape[1]; // Save output size for destination length

        // Weights
        float *data = it->second.data<float>();
        for (size_t offset = 0; offset < it->second.shape[1]; offset += core)
        {
            size_t range = std::min(std::min(core, it->second.shape[1]), it->second.shape[1] - offset);
            for (size_t node = 0; node < it->second.shape[0]; node++)
            {
                for (size_t i = 0; i < range; i++)
                {
                    weight->writeSourceFloat(data[node * it->second.shape[1] + offset + i]);
                }
            }
        }
    }

    // Reset destination
    memset((void *)io->getDestinationAddress(), 0, dst_length * 4);

    if (verbosity_level > 1)
    {
        std::cout << "Loading " << (weight->getCursor() / 4) << " weights" << std::endl;
        std::cout << "Loading " << (config->getCursor() / 8) << " instructions" << std::endl;
    }

    for (size_t n = 0; n < dataset["x"].shape[0]; n++)
    {

        if (verbosity_level == 0)
        {
            bar.progress(n, dataset["x"].shape[0]);
        }

        io->resetCursor();
        // Inputs
        for (size_t i = 0; i < dataset["x"].shape[1]; i++)
        {
            io->writeSourceFloat(input[n * dataset["x"].shape[1] + i]);
        }

        if (verbosity_level > 1)
        {
            std::cout << "Loading " << (io->getCursor() / 4) << " inputs" << std::endl;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Init
        config->reset();
        config->halt();
        config->setInterrupt(true, true, 0);
        config->ready();

        weight->reset();
        weight->halt();
        weight->setInterrupt(true, true, 0);
        weight->ready();

        io->reset();
        io->halt();
        io->setInterrupt(true, true, 0);
        io->ready();

        // Listen
        io->setDestinationAddress(io_dst.addr);
        io->setDestinationLength(dst_length * 4);

        // Send instructions
        config->setSourceAddress(config_src.addr);
        config->setSourceLength(config->getCursor());
        if (verbosity_level > 1)
        {
            std::cout << "Waiting for Instructions MM2S..." << std::endl;
        }
        mem_status = -1;
        do
        {
            status = config->getMM2SStatus();
            if (verbosity_level > 1)
            {
                if (mem_status != status)
                    config->dumpStatus(status);
                mem_status = status;
            }
        } while (
            !(status & 1 << 0) &&
            !(status & 1 << 1) &&
            !(status & 1 << 4) &&
            !(status & 1 << 12) &&
            !(status & 1 << 14));

        // Send input
        io->setSourceAddress(io_src.addr);
        io->setSourceLength(io->getCursor());
        if (verbosity_level > 1)
        {
            std::cout << "Waiting for IO MM2S..." << std::endl;
        }
        mem_status = -1;
        do
        {
            status = io->getMM2SStatus();
            if (verbosity_level > 1)
            {
                if (mem_status != status)
                    io->dumpStatus(status);
                mem_status = status;
            }
        } while (
            !(status & 1 << 0) &&
            !(status & 1 << 1) &&
            !(status & 1 << 4) &&
            !(status & 1 << 12) &&
            !(status & 1 << 14));

        // Send weights
        weight->setSourceAddress(weight_src.addr);
        weight->setSourceLength(weight->getCursor());
        if (verbosity_level > 1)
        {
            std::cout << "Waiting for Weights MM2S..." << std::endl;
        }
        mem_status = -1;
        do
        {
            status = weight->getMM2SStatus();
            if (verbosity_level > 1)
            {
                if (mem_status != status)
                    weight->dumpStatus(status);
                mem_status = status;
            }
        } while (
            !(status & 1 << 0) &&
            !(status & 1 << 1) &&
            !(status & 1 << 4) &&
            !(status & 1 << 12) &&
            !(status & 1 << 14));

        // Wait for output
        if (verbosity_level > 1)
        {
            std::cout << "Waiting for IO S2MM..." << std::endl;
        }
        mem_status = -1;
        do
        {
            status = io->getS2MMStatus();
            if (verbosity_level > 1)
            {
                if (mem_status != status)
                    io->dumpStatus(status);
                mem_status = status;
            }
        } while (
            !(status & 1 << 0) &&
            !(status & 1 << 1) &&
            !(status & 1 << 4) &&
            !(status & 1 << 12) &&
            !(status & 1 << 14));

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        execution_time += duration.count();

        if (verbosity_level > 0)
        {
            std::cout << "Execution time: " << duration.count() << " us" << std::endl;
        }

        // Extract results
        float *fp = (float *)io->getDestinationAddress();
        for (int i = 0; i < dst_length; i++)
            results.push_back(fp[i]);

        // Determine accuracy
        int maxElementIndex = std::max_element(results.begin(), results.end()) - results.begin();
        if (maxElementIndex == (int)output[n])
            correct_classification++;

        if (verbosity_level > 1)
        {
            std::cout << "Result:" << std::endl;
            for (int i = 0; i < dst_length; i++)
                std::cout << "\t" << results[i] << std::endl;
            if (maxElementIndex == (int)output[n])
            {
                std::cout << "Classification is correct: found (#" << (int)output[n] << ")" << std::endl;
            }
            else
            {
                std::cout << "Classification is incorrect: found (#" << maxElementIndex << ") instead of (#" << (int)output[n] << ")" << std::endl;
            }
            system_pause();
        }

        results.clear();
    }

    if (verbosity_level == 0)
    {
        bar.finish();
    }

    std::cout << "Accuracy: " << (float)correct_classification / (float)dataset["x"].shape[0] * 100 << "%" << std::endl;
    std::cout << "Mean execution time: " << (float)execution_time / (float)dataset["x"].shape[0] << " us" << std::endl;

    return 0;
}