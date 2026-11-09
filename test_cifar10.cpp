#include <iostream>
#include <vector>

#include "cifar10_dataset.hpp"
#include "tensor.h"

using namespace std;

int main()
{
    Cifar10Dataset dataset("datasets/cifar10", "train", 0);
    vector<size_t> samples_per_class(10, 0);
    const size_t preview_count = 10;

    cout << "Dataset split: train\n";
    cout << "Dataset size: " << dataset.size() << '\n';

    for (size_t index = 0; index < dataset.size(); ++index) {
        Tensor *image = nullptr;
        size_t label = 0;

        if (!dataset.get(index, &image, &label)) {
            cerr << "Unable to load sample " << index << '\n';
            return 1;
        }

        ++samples_per_class[label];
        if (index < preview_count) {
            cout << "Sample " << index
                 << ": shape=[" << tensor_shape(image)[0]
                 << ", " << tensor_shape(image)[1]
                 << ", " << tensor_shape(image)[2]
                 << "], label=" << label
                 << " (" << dataset.class_name(label) << ")\n";
        }
        tensor_free(image);
    }

    cout << "Samples per class:\n";
    for (size_t label = 0; label < samples_per_class.size(); ++label) {
        cout << "  " << label << " (" << dataset.class_name(label) << "): "
             << samples_per_class[label] << '\n';
    }

    return 0;
}
