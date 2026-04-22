#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <complex>
#include <algorithm>
#include <string>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <iostream>

// extracts MFCCs and f0 from wavs to csv

// g++ scripts\mfcc_calc.cpp -o mfcc_calc.exe
// .\mfcc_calc.exe

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
    double time;     // Added time column
    double log10_f0; // Replaced target_f0 with log10(f0)
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

    if (std::string(header.riff, 4) != "RIFF" || std::string(header.wave, 4) != "WAVE")
        throw std::runtime_error("Invalid WAV file");
    if (header.audio_format != 1)
        throw std::runtime_error("Only PCM WAV supported");
    if (header.bits_per_sample != 16)
        throw std::runtime_error("Only 16-bit WAV supported");

    sample_rate = header.sample_rate;

    char chunk_id[4];
    uint32_t chunk_size;
    while (true)
    {
        file.read(chunk_id, 4);
        file.read(reinterpret_cast<char *>(&chunk_size), 4);
        if (std::string(chunk_id, 4) == "data")
            break;
        file.seekg(chunk_size, std::ios::cur);
    }

    int num_samples = chunk_size / (header.bits_per_sample / 8);
    std::vector<int16_t> buffer(num_samples);
    file.read(reinterpret_cast<char *>(buffer.data()), chunk_size);

    std::vector<double> signal;
    if (header.num_channels == 1)
    {
        signal.resize(num_samples);
        for (int i = 0; i < num_samples; i++)
            signal[i] = buffer[i] / 32768.0;
    }
    else if (header.num_channels == 2)
    {
        int mono_samples = num_samples / 2;
        signal.resize(mono_samples);
        for (int i = 0; i < mono_samples; i++)
        {
            int16_t left = buffer[2 * i];
            int16_t right = buffer[2 * i + 1];
            signal[i] = (left + right) / 65536.0;
        }
    }
    else
        throw std::runtime_error("Unsupported channel count");

    return signal;
}

int next_pow2(int n)
{
    int p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

void apply_hamming(std::vector<double> &frame)
{
    int N = frame.size();
    for (int n = 0; n < N; n++)
    {
        // On garde votre Hanning, qui correspond au defaut de Librosa
        frame[n] *= 0.5 * (1 - cos(2 * M_PI * n / (N - 1)));
    }
}

using Complex = std::complex<double>;

void fft_iterative(std::vector<Complex> &a)
{
    int n = a.size();
    for (int i = 1, j = 0; i < n; i++)
    {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j |= bit;
        if (i < j)
            std::swap(a[i], a[j]);
    }
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

std::vector<double> power_spectrum_rt(const std::vector<double> &frame, int NFFT)
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

// --- YIN Algorithm for librosa equivalent f0 tracking ---
double compute_yin(const std::vector<double> &frame, int sr, double fmin = 50.0, double fmax = 500.0, double threshold = 0.1)
{
    int t_min = sr / fmax;
    int t_max = sr / fmin;
    int N = frame.size();
    int W = N - t_max;

    if (W <= 0)
        return 0.0;

    // 1. Difference function
    std::vector<double> df(t_max + 1, 0.0);
    for (int t = 1; t <= t_max; t++)
    {
        for (int i = 0; i < W; i++)
        {
            double diff = frame[i] - frame[i + t];
            df[t] += diff * diff;
        }
    }

    // 2. Cumulative mean normalized difference function (CMNDF)
    std::vector<double> cmndf(t_max + 1, 1.0);
    double running_sum = 0.0;
    for (int t = 1; t <= t_max; t++)
    {
        running_sum += df[t];
        cmndf[t] = df[t] * t / (running_sum + 1e-12);
    }

    // 3. Absolute thresholding to find early minimum
    int tau_estimate = -1;
    for (int t = t_min; t <= t_max; t++)
    {
        if (cmndf[t] < threshold)
        {
            while (t + 1 <= t_max && cmndf[t + 1] < cmndf[t])
            {
                t++;
            }
            tau_estimate = t;
            break;
        }
    }

    // 4. Global minimum fallback if no trough goes below threshold
    if (tau_estimate == -1)
    {
        tau_estimate = t_min;
        double min_val = cmndf[t_min];
        for (int t = t_min + 1; t <= t_max; t++)
        {
            if (cmndf[t] < min_val)
            {
                min_val = cmndf[t];
                tau_estimate = t;
            }
        }
    }

    // 5. Parabolic interpolation for smoother tracking
    double peak = tau_estimate;
    if (tau_estimate > 0 && tau_estimate < t_max)
    {
        double s0 = cmndf[tau_estimate - 1];
        double s1 = cmndf[tau_estimate];
        double s2 = cmndf[tau_estimate + 1];
        double denom = s0 - 2 * s1 + s2;
        if (denom != 0.0)
        {
            peak = tau_estimate + (s0 - s2) / (2.0 * denom);
        }
    }

    return peak > 0 ? sr / peak : 0.0;
}

// --- MODIFICATION 1 : Échelle de Slaney (Défaut Librosa) au lieu de HTK ---
double hz_to_mel_slaney(double f)
{
    double min_log_hz = 1000.0;
    double min_log_mel = 15.0;
    double logstep = log(6.4) / 27.0;

    if (f >= min_log_hz)
        return min_log_mel + log(f / min_log_hz) / logstep;
    return f / (200.0 / 3.0);
}

double mel_to_hz_slaney(double m)
{
    double min_log_hz = 1000.0;
    double min_log_mel = 15.0;
    double logstep = log(6.4) / 27.0;

    if (m >= min_log_mel)
        return min_log_hz * exp(logstep * (m - min_log_mel));
    return m * (200.0 / 3.0);
}

// --- MODIFICATION 2 : Normalisation de surface du filterbank ---
std::vector<std::vector<double>> mel_filterbank(int nfilt, int NFFT, int sr)
{
    int num_bins = NFFT / 2 + 1;
    double mel_min = hz_to_mel_slaney(0);
    double mel_max = hz_to_mel_slaney(sr / 2);

    std::vector<double> mel_points(nfilt + 2);
    for (int i = 0; i < nfilt + 2; i++)
    {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (nfilt + 1);
    }

    std::vector<int> bins(nfilt + 2);
    for (int i = 0; i < nfilt + 2; i++)
    {
        double hz = mel_to_hz_slaney(mel_points[i]);
        bins[i] = floor((NFFT + 1) * hz / sr);
    }

    std::vector<std::vector<double>> fbank(nfilt, std::vector<double>(num_bins, 0.0));

    for (int m = 1; m <= nfilt; m++)
    {
        double enorm = 2.0 / (mel_to_hz_slaney(mel_points[m + 1]) - mel_to_hz_slaney(mel_points[m - 1]));

        for (int k = bins[m - 1]; k < bins[m]; k++)
        {
            fbank[m - 1][k] = ((k - bins[m - 1]) / double(bins[m] - bins[m - 1] + 1e-12)) * enorm;
        }
        for (int k = bins[m]; k < bins[m + 1]; k++)
        {
            fbank[m - 1][k] = ((bins[m + 1] - k) / double(bins[m + 1] - bins[m] + 1e-12)) * enorm;
        }
    }
    return fbank;
}

std::vector<double> dct(const std::vector<double> &input, int num_ceps)
{
    int N = input.size();
    std::vector<double> out(num_ceps, 0.0);
    double scale0 = sqrt(1.0 / N);
    double scale = sqrt(2.0 / N);

    for (int k = 0; k < num_ceps; k++)
    {
        for (int n = 0; n < N; n++)
        {
            out[k] += input[n] * cos(M_PI * k * (n + 0.5) / N);
        }
        out[k] *= (k == 0) ? scale0 : scale;
    }
    return out;
}

std::vector<AudioFrame> mfcc_calc(const std::string &wav_path)
{
    std::vector<AudioFrame> data;
    int sr = 0;
    std::vector<double> signal = load_wav(wav_path, sr);
    if (signal.empty())
        return data;

    int frame_len = int(0.025 * sr);
    int hop = int(0.01 * sr);
    int NFFT = 1;
    while (NFFT < frame_len)
        NFFT <<= 1;

    int nfilt = 128;
    int num_ceps = 13;

    auto fbank = mel_filterbank(nfilt, NFFT, sr);

    std::string filename = wav_path;
    size_t pos = filename.find_last_of("/\\");
    if (pos != std::string::npos)
        filename = filename.substr(pos + 1);

    int num_frames = (signal.size() - frame_len) / hop + 1;

    std::vector<double> frame(frame_len);
    std::vector<std::vector<double>> all_mels;
    std::vector<double> f0_track(num_frames, 0.0);
    double max_mel = -1e10;

    // Passe 1 : Spectrogramme Mel, Max Global, et f0
    for (int i = 0; i < num_frames; i++)
    {
        for (int j = 0; j < frame_len; j++)
        {
            frame[j] = signal[i * hop + j];
        }

        apply_hamming(frame);
        auto power = power_spectrum_rt(frame, NFFT);

        std::vector<double> mel(nfilt, 0.0);
        for (int m = 0; m < nfilt; m++)
        {
            for (size_t k = 0; k < power.size(); k++)
            {
                mel[m] += power[k] * fbank[m][k];
            }
            mel[m] = 10.0 * log10(std::max(mel[m], 1e-10));
            if (mel[m] > max_mel)
                max_mel = mel[m];
        }
        all_mels.push_back(mel);

        // --- librosa.yin Equivalent Extraction ---
        std::vector<double> yin_frame(NFFT, 0.0);
        for (int j = 0; j < NFFT && (i * hop + j) < signal.size(); j++)
        {
            yin_frame[j] = signal[i * hop + j];
        }

        double f0 = compute_yin(yin_frame, sr, 50.0, 500.0);
        f0_track[i] = (f0 > 0.0) ? std::log10(f0) : 0.0;
    }

    // Passe 2 : Limite de bruit (Top DB) et DCT
    double top_db = 80.0;
    for (int i = 0; i < num_frames; i++)
    {
        for (int m = 0; m < nfilt; m++)
        {
            all_mels[i][m] = std::max(all_mels[i][m], max_mel - top_db);
        }

        auto cep = dct(all_mels[i], num_ceps);

        AudioFrame af;
        af.filename = filename;
        af.frame_idx = i;
        af.time = 0.0;             // Enforced 0.0 per instructions
        af.log10_f0 = f0_track[i]; // Stores the log10 equivalent computed above

        for (int j = 0; j < 13; j++)
        {
            af.mfcc[j] = cep[j];
        }
        data.push_back(af);
    }

    return data;
}

void write_csv(const std::string &output_path, const std::vector<AudioFrame> &data, bool write_header = true)
{
    std::ifstream infile(output_path);
    bool file_exists = infile.good();
    infile.close();

    std::ofstream file(output_path, std::ios::out);
    if (!file)
        throw std::runtime_error("Failed to open CSV file");

    if (!file_exists && write_header)
    {
        // Exact header sequence defined in prompt
        file << "filename,frame_index,time,log10(f0),mfcc_0,mfcc_1,mfcc_2,mfcc_3,mfcc_4,mfcc_5,mfcc_6,mfcc_7,mfcc_8,mfcc_9,mfcc_10,mfcc_11,mfcc_12\n";
    }

    for (const auto &frame : data)
    {
        file << frame.filename << ","
             << frame.frame_idx << ","
             << frame.time << ","
             << frame.log10_f0;

        for (int i = 0; i < 13; i++)
            file << "," << frame.mfcc[i];

        file << "\n";
    }
    file.close();
}

int main()
{
    auto frames = mfcc_calc("C:\\Users\\alexa\\OneDrive\\Desktop\\Stage GIPSA-lab\\LJSpeech-1.1\\LJSpeech-1.1\\big_wavs\\LJ001-0021.wav");
    write_csv("C:\\Users\\alexa\\OneDrive\\Desktop\\Stage GIPSA-lab\\C++ audio-data_extraction\\cpp_mfcc_extraction-1.csv", frames);

    return 0;
}