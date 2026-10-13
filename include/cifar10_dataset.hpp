#ifndef CIFAR10_DATASET_HPP
#define CIFAR10_DATASET_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct Tensor;

class Cifar10Dataset {
public:
    Cifar10Dataset(
        const std::string &root,
        const std::string &split,
        std::size_t max_samples = 0);

    std::size_t size() const;
    const std::string &class_name(std::size_t label) const;
    void shuffle(std::uint32_t seed);

    /* Load one normalized image tensor with shape [32, 32, 3]. */
    bool get(std::size_t index, Tensor **image, std::size_t *label) const;

private:
    struct Sample {
        std::string path;
        std::size_t label;
    };

    std::vector<Sample> samples_;
    std::vector<std::string> class_names_;
};

#endif /* CIFAR10_DATASET_HPP */
