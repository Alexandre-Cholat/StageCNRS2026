#define EIGEN_STACK_ALLOCATION_LIMIT 0

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <RTNeural/RTNeural.h>

// ==========================================
// DATA STRUCTURES
// ==========================================
struct AudioFrame
{
    std::string filename;
    int frame_idx;
    double target_f0;
    double mfcc[13];
};

// ==========================================
// CSV PARSER FUNCTION
// ==========================================
std::vector<AudioFrame> readCSV(std::string filePath)
{
    std::vector<AudioFrame> data;
    std::ifstream file(filePath);

    if (!file.is_open())
    {
        std::cerr << "Could not open the CSV file!" << std::endl;
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

bool loadNormalizationStats(const std::string &statsPath, float (&mfcc_mean)[13], float (&mfcc_std)[13], float &f0_mean, float &f0_std)
{
    std::ifstream statsFile(statsPath);
    if (!statsFile.is_open())
    {
        std::cerr << "Could not open normalization stats file: " << statsPath << std::endl;
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

// COMPILE-TIME API

using F0ModelType = RTNeural::ModelT<float, 13, 1,
    RTNeural::DenseT<float, 13, 256>,        // 0
    RTNeural::ReLuActivationT<float, 256>,   // 1
    RTNeural::DenseT<float, 256, 128>,       // 2
    RTNeural::ReLuActivationT<float, 128>,   // 3
    RTNeural::DenseT<float, 128, 64>,        // 4
    RTNeural::ReLuActivationT<float, 64>,    // 5
    RTNeural::DenseT<float, 64, 32>,         // 6
    RTNeural::ReLuActivationT<float, 32>,    // 7
    
    // Conv1DT with 17 frame buffer (introduces 80ms latency)
    RTNeural::Conv1DT<float, 32, 64, 17, 1>, // 8
    RTNeural::ReLuActivationT<float, 64>,    // 9
    
    RTNeural::DenseT<float, 64, 64>,         // 10
    RTNeural::ReLuActivationT<float, 64>,    // 11
    RTNeural::DenseT<float, 64, 32>,         // 12
    RTNeural::ReLuActivationT<float, 32>,    // 13
    RTNeural::DenseT<float, 32, 16>,         // 14
    RTNeural::ReLuActivationT<float, 16>,    // 15
    RTNeural::DenseT<float, 16, 1>           // 16
>;

// ==========================================
// MAIN INFERENCE LOOP
// ==========================================
int main() {
    try {
        std::cout << "Initializing RTNeural Compile-Time Model..." << std::endl;
        alignas(32) F0ModelType f0_model;

        // 1. Load the raw PyTorch state_dict JSON weights
        std::ifstream jsonStream("C:\\Users\\alexa\\OneDrive\\Desktop\\StageCNRS2026\\model7_weights.json");
        if (!jsonStream.is_open()) {
            std::cerr << "Failed to find model weights!" << std::endl;
            return 1;
        }

        nlohmann::json modelJson;
        jsonStream >> modelJson;

        std::cout << "Loading weights into RTNeural layers..." << std::endl;
        
        // 2. Load Weights into the specific indices
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

        // 3. Load normalization stats
        float mfcc_mean[13]{}; float mfcc_std[13]{}; float f0_mean = 0.0f; float f0_std = 1.0f;
        if (!loadNormalizationStats("C:\\Users\\alexa\\OneDrive\\Desktop\\StageCNRS2026\\normalisationStats.json", mfcc_mean, mfcc_std, f0_mean, f0_std)) return 1;

        // 4. Load the data
        std::vector<AudioFrame> frames = readCSV("C:\\Users\\alexa\\OneDrive\\Desktop\\Stage GIPSA-lab\\audio-data_extraction\\10_test_wavs_MFCC_f0_extraction.csv");
        if (frames.empty()) return 1;

        std::cout << "Starting inference on " << frames.size() << " frames..." << std::endl;

        // 5. Run inference
        alignas(32) float input_array[13];
        float squared_error_sum = 0.0;
        float absolute_error_sum = 0.0;
        for (const auto &frame : frames) {

            // Normalization
            for (int i = 0; i < 13; ++i) {
                input_array[i] = static_cast<float>((frame.mfcc[i] - mfcc_mean[i]) / mfcc_std[i]);
            }
            
            // Forward pass:
            // RTNeural automatically maintains an internal "state" buffer. 
            // It remembers the previous frames and uses them to calculate the 
            // temporal context for the current prediction
            float normalized_f0_pred = f0_model.forward(input_array);

            // remove normalisation
            float f0_pred = (normalized_f0_pred * f0_std) + f0_mean;
            
            std::cout << "Frame: " << frame.frame_idx
                        << " | Pred F0 : " << f0_pred
                        << " | Target F0 : " << frame.target_f0 << std::endl;
            
            squared_error_sum += std::pow(f0_pred - static_cast<float>(frame.target_f0), 2);
            absolute_error_sum += std::abs(f0_pred - static_cast<float>(frame.target_f0));
        }

        float mse = squared_error_sum / frames.size();
        float mae = absolute_error_sum / frames.size();

        std::cout << "Inference complete." << std::endl;
        std::cout << "Mean Squared Error: " << mse << std::endl;
        std::cout << "Mean Absolute Error: " << mae << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nFATAL CRASH DETECTED: " << e.what() << std::endl;
        return 1;
    }
}