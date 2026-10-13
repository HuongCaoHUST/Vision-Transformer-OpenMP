#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <omp.h>

#include "cifar10_dataset.hpp"
#include "tensor.h"
#include "vision_transformer.h"

using namespace std;

int main(int argc, char **argv)
{
    const size_t epochs = argc > 1 ? strtoul(argv[1], nullptr, 10) : 1;
    const size_t max_samples = argc > 2 ? strtoul(argv[2], nullptr, 10) : 100;
    Cifar10Dataset dataset("datasets/cifar10", "train", max_samples);
    VisionTransformer *model;
    float learning_rate = 0.01f;

    if (dataset.size() == 0) {
        cerr << "The CIFAR-10 training dataset is empty\n";
        return EXIT_FAILURE;
    }

    model = vision_transformer_create(32, 32, 8, 3, 64, 4, 256, 2, 10);
    if (model == nullptr) {
        cerr << "Unable to create Vision Transformer\n";
        return EXIT_FAILURE;
    }

    cout << "Training samples: " << dataset.size() << '\n';
    cout << "Epochs: " << epochs << '\n';
    cout << "OpenMP threads: " << omp_get_max_threads() << '\n';
    cout << "Trainable parameters: classification head only\n";

    const double training_start = omp_get_wtime();
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        const double epoch_start = omp_get_wtime();
        float total_loss = 0.0f;
        size_t successful_samples = 0;
        const size_t sample_count = dataset.size();
        vector<Tensor *> features(sample_count, nullptr);
        vector<size_t> labels(sample_count, 0);
        vector<unsigned char> valid_samples(sample_count, 0);

        dataset.shuffle(static_cast<uint32_t>(epoch + 1));
        /* Feature extraction is read-only and can safely run in parallel. */
        #pragma omp parallel for schedule(dynamic)
        for (long long index = 0; index < static_cast<long long>(sample_count); ++index) {
            Tensor *image = nullptr;
            size_t label = 0;

            if (!dataset.get(static_cast<size_t>(index), &image, &label)) {
                continue;
            }

            features[static_cast<size_t>(index)] =
                vision_transformer_forward_features(model, image);
            labels[static_cast<size_t>(index)] = label;
            valid_samples[static_cast<size_t>(index)] = static_cast<unsigned char>(
                features[static_cast<size_t>(index)] != nullptr);
            tensor_free(image);
        }

        /* Classifier updates remain sequential because they modify shared weights. */
        for (size_t index = 0; index < sample_count; ++index) {
            if (!valid_samples[index]) {
                continue;
            }

            float loss = 0.0f;
            if (vision_transformer_train_classifier_step(
                    model, features[index], labels[index], learning_rate, &loss) != TENSOR_OK) {
                tensor_free(features[index]);
                features[index] = nullptr;
                continue;
            }

            total_loss += loss;
            ++successful_samples;
            tensor_free(features[index]);
            features[index] = nullptr;
        }

        const float average_loss = successful_samples == 0
            ? 0.0f : total_loss / static_cast<float>(successful_samples);
        const double epoch_seconds = omp_get_wtime() - epoch_start;
        cout << "Epoch " << (epoch + 1)
             << "/" << epochs
             << " - average loss: "
             << fixed << setprecision(6) << average_loss
             << " - time: " << epoch_seconds << " s"
             << " - samples/s: "
             << (epoch_seconds > 0.0
                     ? static_cast<double>(successful_samples) / epoch_seconds
                     : 0.0)
             << '\n';
    }

    const double training_seconds = omp_get_wtime() - training_start;
    cout << "Total training time: " << fixed << setprecision(6)
         << training_seconds << " s\n";
    vision_transformer_free(model);
    return EXIT_SUCCESS;
}
