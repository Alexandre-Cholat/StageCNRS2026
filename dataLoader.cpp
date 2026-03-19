#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// This struct is like a Class in Java or a Dict in Python
struct AudioFrame
{
    std::string filename;
    int frame_idx;
    double target_f0;
    double mfcc[13];
};

std::vector<AudioFrame> readCSV(std::string filePath)
{
    std::vector<AudioFrame> data;
    std::ifstream file(filePath);

    if (!file.is_open())
    {
        std::cerr << "Could not open the file!" << std::endl;
        return data;
    }

    std::string line;
    // Skip the header row
    std::getline(file, line);

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;
        AudioFrame frame;

        // 1. Get filename
        std::getline(ss, frame.filename, ',');

        // 2. Get frame_ind
        std::getline(ss, value, ',');
        frame.frame_idx = std::stoi(value);

        // 3. skip time
        std::getline(ss, value, ',');

        // 4. Get log10(f0)
        std::getline(ss, value, ',');
        frame.target_f0 = std::stod(value);

        // 5. Get MFCCs 0-12
        for (int i = 0; i < 13; ++i)
        {
            std::getline(ss, value, ',');
            frame.mfcc[i] = std::stod(value);
        }

        data.push_back(frame);
    }

    file.close();
    return data;
}

int main()
{
    std::vector<AudioFrame> audioData = readCSV("C:\\Users\\alexa\\OneDrive\\Desktop\\Stage GIPSA-lab\\audio-data_extraction\\10_test_wavs_MFCC_f0_extraction.csv");

    // Quick test: Print the first MFCC of the first row
    if (!audioData.empty())
    {
        std::cout << "File: " << audioData[1].filename << " | MFCC_0: " << audioData[1].mfcc[0] << std::endl;
    }

    return 0;
}