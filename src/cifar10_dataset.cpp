#include "cifar10_dataset.hpp"

#include <algorithm>
#include <filesystem>
#include <random>

#include <opencv2/opencv.hpp>

#include "tensor.h"

namespace fs = std::filesystem;

Cifar10Dataset::Cifar10Dataset(
    const std::string &root,
    const std::string &split,
    std::size_t max_samples)
    : class_names_{
          "airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"}
{
    const fs::path split_path = fs::path(root) / split;

    for (std::size_t label = 0; label < class_names_.size(); ++label) {
        const fs::path class_path = split_path / class_names_[label];
        if (!fs::exists(class_path) || !fs::is_directory(class_path)) {
            continue;
        }

        for (const fs::directory_entry &entry : fs::directory_iterator(class_path)) {
            if (!entry.is_regular_file()) {
                continue;
            }

            const std::string extension = entry.path().extension().string();
            if (extension == ".png" || extension == ".jpg" || extension == ".jpeg") {
                samples_.push_back({entry.path().string(), label});
            }
        }
    }

    std::sort(samples_.begin(), samples_.end(),
              [](const Sample &left, const Sample &right) {
                  return left.path < right.path;
              });

    if (max_samples != 0 && samples_.size() > max_samples) {
        std::mt19937 generator(0xC1FA10u);
        std::shuffle(samples_.begin(), samples_.end(), generator);
        samples_.resize(max_samples);
    }
}

std::size_t Cifar10Dataset::size() const
{
    return samples_.size();
}

const std::string &Cifar10Dataset::class_name(std::size_t label) const
{
    return class_names_.at(label);
}

void Cifar10Dataset::shuffle(std::uint32_t seed)
{
    std::mt19937 generator(seed);
    std::shuffle(samples_.begin(), samples_.end(), generator);
}

bool Cifar10Dataset::get(
    std::size_t index,
    Tensor **image,
    std::size_t *label) const
{
    cv::Mat loaded;
    std::size_t shape[3];
    Tensor *tensor;
    float *data;

    if (image == nullptr || label == nullptr || index >= samples_.size()) {
        return false;
    }

    loaded = cv::imread(samples_[index].path, cv::IMREAD_COLOR);
    if (loaded.empty() || loaded.rows != 32 || loaded.cols != 32 || loaded.channels() != 3) {
        return false;
    }

    shape[0] = 32;
    shape[1] = 32;
    shape[2] = 3;
    tensor = tensor_zeros(shape, 3);
    if (tensor == nullptr) {
        return false;
    }

    data = tensor_data_mut(tensor);
    for (int row = 0; row < loaded.rows; ++row) {
        for (int column = 0; column < loaded.cols; ++column) {
            const cv::Vec3b pixel = loaded.at<cv::Vec3b>(row, column);
            const std::size_t offset =
                (static_cast<std::size_t>(row) * loaded.cols
                 + static_cast<std::size_t>(column)) * 3;

            for (int channel = 0; channel < 3; ++channel) {
                data[offset + static_cast<std::size_t>(channel)] =
                    static_cast<float>(pixel[channel]) / 255.0f;
            }
        }
    }

    *image = tensor;
    *label = samples_[index].label;
    return true;
}
