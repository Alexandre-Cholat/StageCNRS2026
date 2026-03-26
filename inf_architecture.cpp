#define EIGEN_STACK_ALLOCATION_LIMIT 0

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <RTNeural/RTNeural.h>

#include <chrono>  // For fetching system time
#include <ctime>   // For converting time structures
#include <iomanip> // For formatting the time (std::put_time)

// DATA STRUCTURES
struct AudioFrame
{
    std::string filename;
    int frame_idx;
    double target_f0;
    double mfcc[13];
};

// CSV PARSER
std::vector<AudioFrame> readCSV(std::string filePath)
{
    std::vector<AudioFrame> data;
    std::ifstream file(filePath);

    if (!file.is_open())
    {
        std::cerr << "Could not open CSV. " << std::endl;
        return data;
    }

    std::string line, value;
    std::getline(file, line); // Skip header

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        AudioFrame frame;

        std::getline(ss, frame.filename, ',');

        if (std::getline(ss, value, ','))
            frame.frame_idx = std::stoi(value);

        std::getline(ss, value, ','); // Skip time column

        if (std::getline(ss, value, ','))
            frame.target_f0 = std::stod(value);

        for (int i = 0; i < 13; ++i)
        {
            if (std::getline(ss, value, ','))
            {
                frame.mfcc[i] = std::stod(value);
            }
        }
        data.push_back(frame);
    }
    return data;
}

// dynamic NORMALIZATION STATS LOADER
bool loadNormalizationStats(const std::string &statsPath, float (&mfcc_mean)[13], float (&mfcc_std)[13], float &f0_mean, float &f0_std)
{
    std::ifstream statsFile(statsPath);
    if (!statsFile.is_open())
    {
        std::cerr << "Could not open normalization stats: " << statsPath << std::endl;
        return false;
    }

    try
    {
        nlohmann::json statsJson;
        statsFile >> statsJson;

        if (!statsJson.contains("mfcc_mean") || !statsJson.contains("mfcc_std") || !statsJson.contains("f0_mean") || !statsJson.contains("f0_std"))
        {
            std::cerr << "normalisationStats.json is missing required keys." << std::endl;
            return false;
        }

        const auto mfccMeanValues = statsJson.at("mfcc_mean").get<std::vector<float>>();
        const auto mfccStdValues = statsJson.at("mfcc_std").get<std::vector<float>>();

        if (mfccMeanValues.size() != 13 || mfccStdValues.size() != 13)
        {
            std::cerr << "Expected 13 MFCC mean/std values in normalisationStats.json." << std::endl;
            return false;
        }

        for (int i = 0; i < 13; ++i)
        {
            mfcc_mean[i] = mfccMeanValues[(size_t)i];
            mfcc_std[i] = mfccStdValues[(size_t)i];
        }

        f0_mean = statsJson.at("f0_mean").get<float>();
        f0_std = statsJson.at("f0_std").get<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to parse normalization stats: " << e.what() << std::endl;
        return false;
    }

    return true;
}

// COMPILE-TIME API setup
using F0ModelType = RTNeural::ModelT<float, 13, 1,
                                     RTNeural::DenseT<float, 13, 256>,      // 0
                                     RTNeural::ReLuActivationT<float, 256>, // 1
                                     RTNeural::DenseT<float, 256, 128>,     // 2
                                     RTNeural::ReLuActivationT<float, 128>, // 3
                                     RTNeural::DenseT<float, 128, 64>,      // 4
                                     RTNeural::ReLuActivationT<float, 64>,  // 5
                                     RTNeural::DenseT<float, 64, 32>,       // 6
                                     RTNeural::ReLuActivationT<float, 32>,  // 7

                                     // Conv1DT with 17 frame buffer (introduces 80ms latency)
                                     RTNeural::Conv1DT<float, 32, 64, 17, 1>, // 8
                                     RTNeural::ReLuActivationT<float, 64>,    // 9

                                     RTNeural::DenseT<float, 64, 64>,      // 10
                                     RTNeural::ReLuActivationT<float, 64>, // 11
                                     RTNeural::DenseT<float, 64, 32>,      // 12
                                     RTNeural::ReLuActivationT<float, 32>, // 13
                                     RTNeural::DenseT<float, 32, 16>,      // 14
                                     RTNeural::ReLuActivationT<float, 16>, // 15
                                     RTNeural::DenseT<float, 16, 1>        // 16
                                     >;

// f0 INFERENCE LOOP
int main()
{
    try
    {
        std::cout << "Initializing RTNeural Compile-Time Model..." << std::endl;
        alignas(32) F0ModelType f0_model;

        // load weights
        std::ifstream jsonStream("C:\\Users\\alexa\\OneDrive\\Desktop\\StageCNRS2026\\model7_weights.json");
        if (!jsonStream.is_open())
        {
            std::cerr << "Failed to find model weights!" << std::endl;
            return 1;
        }
        nlohmann::json modelJson;
        jsonStream >> modelJson;
        std::cout << "Loading weights into RTNeural layers..." << std::endl;

        // load weights into corresponding indices
        RTNeural::torch_helpers::loadDense<float>(modelJson, "encoder.network.0.", f0_model.get<0>());
        RTNeural::torch_helpers::loadDense<float>(modelJson, "encoder.network.2.", f0_model.get<2>());
        RTNeural::torch_helpers::loadDense<float>(modelJson, "encoder.network.4.", f0_model.get<4>());
        RTNeural::torch_helpers::loadDense<float>(modelJson, "encoder.network.6.", f0_model.get<6>());

        RTNeural::torch_helpers::loadConv1D<float>(modelJson, "decoder.conv1d.", f0_model.get<8>());

        RTNeural::torch_helpers::loadDense<float>(modelJson, "decoder.fc_layers.0.", f0_model.get<10>());
        RTNeural::torch_helpers::loadDense<float>(modelJson, "decoder.fc_layers.3.", f0_model.get<12>());
        RTNeural::torch_helpers::loadDense<float>(modelJson, "decoder.fc_layers.5.", f0_model.get<14>());
        RTNeural::torch_helpers::loadDense<float>(modelJson, "decoder.fc_layers.7.", f0_model.get<16>());

        f0_model.reset();
        std::cout << "Weights loaded successfully!" << std::endl;

        // load normalization stats
        float mfcc_mean[13]{};
        float mfcc_std[13]{};
        float f0_mean = 0.0f;
        float f0_std = 1.0f;
        if (!loadNormalizationStats("C:\\Users\\alexa\\OneDrive\\Desktop\\StageCNRS2026\\normalisationStats.json", mfcc_mean, mfcc_std, f0_mean, f0_std))
            return 1;

        // load data pipeline
        std::vector<AudioFrame> frames = readCSV("C:\\Users\\alexa\\OneDrive\\Desktop\\Stage GIPSA-lab\\audio-data_extraction\\10_test_wavs_MFCC_f0_extraction.csv");
        if (frames.empty())
            return 1;

        std::cout << "Starting inference on " << frames.size() << " frames..." << std::endl;

        // create output CSV for predictions
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);

        // build the filename
        std::stringstream filename_ss;
        filename_ss << "real-time-predictions_"
                    << std::put_time(std::localtime(&now_c), "%Y-%m-%d_%H-%M-%S")
                    << ".csv";

        std::string filename = filename_ss.str();
        std::ofstream csv_file(filename);
        // Write the CSV header
        csv_file << "frame_idx,f0_pred,target_f0\n";

        // run inference
        alignas(32) float input_array[13];
        float squared_error_sum = 0.0;
        float absolute_error_sum = 0.0;
        for (const auto &frame : frames)
        {

            // Normalization
            for (int i = 0; i < 13; ++i)
            {
                input_array[i] = static_cast<float>((frame.mfcc[i] - mfcc_mean[i]) / mfcc_std[i]);
            }

            // Forward pass:
            // RTNeural has an internal "state" buffer.
            // - remembers previous frames and uses them to calculate current prediction
            float normalized_f0_pred = f0_model.forward(input_array);

            // remove normalisation
            float f0_pred = (normalized_f0_pred * f0_std) + f0_mean;

            //terminal output
            std::cout << "Frame: " << frame.frame_idx
                      << " | Pred F0 : " << f0_pred
                      << " | Target F0 : " << frame.target_f0 << std::endl;

            // write to CSV
            csv_file << frame.frame_idx << "," 
             << f0_pred << "," 
             << frame.target_f0 << "\n";

            // ne pas caluler les erreurs ici. sauvgarder f0 predites en csv et faire analyse en python.
            squared_error_sum += std::pow(f0_pred - static_cast<float>(frame.target_f0), 2);
            absolute_error_sum += std::abs(f0_pred - static_cast<float>(frame.target_f0));
        }
        csv_file.close();

        float mse = squared_error_sum / frames.size();
        float mae = absolute_error_sum / frames.size();

        std::cout << "Inference complete." << std::endl;
        std::cout << "Mean Squared Error: " << mse << std::endl;
        std::cout << "RMSE: " << std::sqrt(mse) << std::endl;
        std::cout << "Mean Absolute Error: " << mae << std::endl;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nFATAL CRASH DETECTED: " << e.what() << std::endl;
        return 1;
    }
}