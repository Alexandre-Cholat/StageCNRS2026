#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <complex>
#include <algorithm>
#include <string>
#include <fstream>
#include <cstdint>
#include <stdexcept>

// extracts MFCCs and f0 from wavs to csv

struct WavHeader
{
    char riff[4]; // "RIFF"
    uint32_t chunk_size;
    char wave[4]; // "WAVE"
    char fmt[4];  // "fmt "
    uint32_t subchunk1_size;
    uint16_t audio_format; // PCM = 1
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};

struct AudioFrame
{
    std::string filename;
    int frame_idx;
    double target_f0;
    double mfcc[13];
};

// wavs loader

std::vector<double> load_wav(const std::string &filename, int &sample_rate)
{

    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to open WAV file");

    WavHeader header;
    file.read(reinterpret_cast<char *>(&header), sizeof(WavHeader));

    // --- Basic validation ---
    if (std::string(header.riff, 4) != "RIFF" ||
        std::string(header.wave, 4) != "WAVE")
    {
        throw std::runtime_error("Invalid WAV file");
    }

    if (header.audio_format != 1)
    {
        throw std::runtime_error("Only PCM WAV supported");
    }

    if (header.bits_per_sample != 16)
    {
        throw std::runtime_error("Only 16-bit WAV supported");
    }

    sample_rate = header.sample_rate;

    // --- Find "data" chunk (skip extra chunks properly) ---
    char chunk_id[4];
    uint32_t chunk_size;

    while (true)
    {
        file.read(chunk_id, 4);
        file.read(reinterpret_cast<char *>(&chunk_size), 4);

        if (std::string(chunk_id, 4) == "data")
        {
            break;
        }

        // Skip unknown chunk
        file.seekg(chunk_size, std::ios::cur);
    }

    // --- Read audio data ---
    int num_samples = chunk_size / (header.bits_per_sample / 8);
    std::vector<int16_t> buffer(num_samples);

    file.read(reinterpret_cast<char *>(buffer.data()), chunk_size);

    // --- Convert to mono double [-1,1] ---
    std::vector<double> signal;

    if (header.num_channels == 1)
    {
        signal.resize(num_samples);
        for (int i = 0; i < num_samples; i++)
        {
            signal[i] = buffer[i] / 32768.0;
        }
    }
    else if (header.num_channels == 2)
    {
        int mono_samples = num_samples / 2;
        signal.resize(mono_samples);

        for (int i = 0; i < mono_samples; i++)
        {
            int16_t left = buffer[2 * i];
            int16_t right = buffer[2 * i + 1];

            signal[i] = (left + right) / 65536.0; // average + normalize
        }
    }
    else
    {
        throw std::runtime_error("Unsupported channel count");
    }

    return signal;
}

// MFCC calc utility functions
int next_pow2(int n)
{
    int p = 1;
    while (p < n)
        p <<= 1;
    return p;
}
// Hanning window
void apply_hamming(std::vector<double> &frame)
{
    int N = frame.size();
    for (int n = 0; n < N; n++)
    {
        // hamming window
        // frame[n] *= 0.54 - 0.46 * cos(2 * M_PI * n / (N - 1));

        // hanning window
        frame[n] *= 0.5 * (1 - cos(2 * M_PI * n / (N - 1)));
    }
}
// FFT (slow)
#include <complex>
using Complex = std::complex<double>;

void fft_iterative(std::vector<Complex> &a)
{
    int n = a.size();

    // Bit reversal
    for (int i = 1, j = 0; i < n; i++)
    {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j |= bit;
        if (i < j)
            std::swap(a[i], a[j]);
    }

    // FFT
    for (int len = 2; len <= n; len <<= 1)
    {
        double angle = -2 * M_PI / len;
        Complex wlen(cos(angle), sin(angle));

        for (int i = 0; i < n; i += len)
        {
            Complex w(1);
            for (int j = 0; j < len / 2; j++)
            {
                Complex u = a[i + j];
                Complex v = a[i + j + len / 2] * w;

                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;

                w *= wlen;
            }
        }
    }
}

// power spectrum
std::vector<double> power_spectrum_rt(
    const std::vector<double> &frame,
    int NFFT)
{
    std::vector<Complex> x(NFFT, 0.0);

    for (size_t i = 0; i < frame.size(); i++)
        x[i] = frame[i];

    fft_iterative(x);

    std::vector<double> power(NFFT / 2 + 1);
    for (int i = 0; i <= NFFT / 2; i++)
    {
        power[i] = std::norm(x[i]);
    }

    return power;
}

// htk mel scale
double hz_to_mel(double f)
{
    return 2595.0 * log10(1.0 + f / 700.0);
}

double mel_to_hz(double m)
{
    return 700.0 * (pow(10.0, m / 2595.0) - 1.0);
}

// mel filterbank
std::vector<std::vector<double>> mel_filterbank(
    int nfilt, int NFFT, int sr)
{
    int num_bins = NFFT / 2 + 1;

    double mel_min = hz_to_mel(0);
    double mel_max = hz_to_mel(sr / 2);

    std::vector<double> mel_points(nfilt + 2);
    for (int i = 0; i < nfilt + 2; i++)
    {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (nfilt + 1);
    }

    std::vector<int> bins(nfilt + 2);
    for (int i = 0; i < nfilt + 2; i++)
    {
        double hz = mel_to_hz(mel_points[i]);
        bins[i] = floor((NFFT + 1) * hz / sr);
    }

    std::vector<std::vector<double>> fbank(nfilt,
                                           std::vector<double>(num_bins, 0.0));

    for (int m = 1; m <= nfilt; m++)
    {
        for (int k = bins[m - 1]; k < bins[m]; k++)
        {
            fbank[m - 1][k] =
                (k - bins[m - 1]) / double(bins[m] - bins[m - 1] + 1e-12);
        }
        for (int k = bins[m]; k < bins[m + 1]; k++)
        {
            fbank[m - 1][k] =
                (bins[m + 1] - k) / double(bins[m + 1] - bins[m] + 1e-12);
        }
    }

    return fbank;
}

// log + mel energy
std::vector<double> apply_mel(
    const std::vector<double> &power,
    const std::vector<std::vector<double>> &fbank)
{
    std::vector<double> out(fbank.size());

    for (int i = 0; i < fbank.size(); i++)
    {
        double sum = 0.0;
        for (int j = 0; j < power.size(); j++)
        {
            sum += power[j] * fbank[i][j];
        }
        out[i] = log(std::max(sum, 1e-10)); // natural log
    }
    return out;
}

// DCT
std::vector<double> dct(
    const std::vector<double> &input,
    int num_ceps)
{
    int N = input.size();
    std::vector<double> out(num_ceps, 0.0);

    double scale0 = sqrt(1.0 / N);
    double scale = sqrt(2.0 / N);

    for (int k = 0; k < num_ceps; k++)
    {
        for (int n = 0; n < N; n++)
        {
            out[k] += input[n] *
                      cos(M_PI * k * (n + 0.5) / N);
        }
        out[k] *= (k == 0) ? scale0 : scale;
    }

    return out;
}

// framing
std::vector<std::vector<double>> frame_signal(
    const std::vector<double> &signal,
    int frame_len,
    int hop)
{
    int num_frames = (signal.size() - frame_len) / hop + 1;

    std::vector<std::vector<double>> frames;

    for (int i = 0; i < num_frames; i++)
    {
        std::vector<double> frame(frame_len);
        for (int j = 0; j < frame_len; j++)
        {
            frame[j] = signal[i * hop + j];
        }
        frames.push_back(frame);
    }

    return frames;
}

// extract MFCCs from signal
std::vector<AudioFrame> mfcc_calc(const std::string &wav_path)
{

    std::vector<AudioFrame> data;

    int sr = 0;
    std::vector<double> signal = load_wav(wav_path, sr);
    if (signal.empty())
        return data;

    // Parameters (matching your Python)
    int frame_len = int(0.025 * sr);
    int hop = int(0.01 * sr);
    int NFFT = 1;
    while (NFFT < frame_len)
        NFFT <<= 1;

    int nfilt = 26;
    int num_ceps = 13;

    auto fbank = mel_filterbank(nfilt, NFFT, sr);

    std::string filename = wav_path;
    size_t pos = filename.find_last_of("/\\");
    if (pos != std::string::npos)
        filename = filename.substr(pos + 1);

    int num_frames = (signal.size() - frame_len) / hop + 1;

    std::vector<double> frame(frame_len);

    for (int i = 0; i < num_frames; i++)
    {

        // --- Copy frame ---
        for (int j = 0; j < frame_len; j++)
        {
            frame[j] = signal[i * hop + j];
        }

        // --- Window ---
        apply_hamming(frame);

        // --- FFT + Power ---
        auto power = power_spectrum_rt(frame, NFFT);

        // --- Mel ---
        std::vector<double> mel(nfilt, 0.0);
        for (int m = 0; m < nfilt; m++)
        {
            for (int k = 0; k < power.size(); k++)
            {
                mel[m] += power[k] * fbank[m][k];
            }
            mel[m] = 10 * log10(std::max(mel[m], 1e-10));
        }

        // --- DCT ---
        auto cep = dct(mel, num_ceps);

        // --- Store ---
        AudioFrame af;
        af.filename = filename;
        af.frame_idx = i;
        af.target_f0 = 0.0;

        for (int j = 0; j < 13; j++)
        {
            af.mfcc[j] = cep[j];
        }

        data.push_back(af);
    }

    return data;
}

// write audio frames to csv
void write_csv(
    const std::string &output_path,
    const std::vector<AudioFrame> &data,
    bool write_header = true)
{
    // Check if file exists
    std::ifstream infile(output_path);
    bool file_exists = infile.good();
    infile.close();

    std::ofstream file(output_path, std::ios::app);
    if (!file)
        throw std::runtime_error("Failed to open CSV file");

    // Write header only if needed
    if (!file_exists && write_header)
    {
        file << "filename,frame_idx,target_f0";
        for (int i = 0; i < 13; i++)
        {
            file << ",mfcc_" << i;
        }
        file << "\n";
    }

    // Write data
    for (const auto &frame : data)
    {
        file << frame.filename << ",";
        file << frame.frame_idx << ",";

        for (int i = 0; i < 13; i++)
        {
            file << "," << frame.mfcc[i];
        }

        file << "\n";
    }

    file.close();
}

int main()
{

    auto frames = mfcc_calc("C:\\Users\\alexa\\OneDrive\\Desktop\\Stage GIPSA-lab\\LJSpeech-1.1\\LJSpeech-1.1\\big_wavs\\LJ001-0021.wav");
    write_csv("C:\\Users\\alexa\\OneDrive\\Desktop\\Stage GIPSA-lab\\C++ audio-data_extraction\\cpp_mfcc_extraction.csv", frames);

    return 0;
}