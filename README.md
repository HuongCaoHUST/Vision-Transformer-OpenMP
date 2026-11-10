# Vision Transformer OpenMP

Triển khai Vision Transformer (ViT) bằng C/C++, OpenCV và OpenMP. Pipeline đọc CIFAR-10 dạng thư mục, huấn luyện toàn bộ ViT với batch size = 1 và đánh giá trên train/test sau mỗi epoch.

## Yêu cầu

- Linux hoặc môi trường có `bash`
- GCC/G++, GNU Make
- OpenMP
- OpenCV 4 và `opencv4.pc`

Trên Ubuntu/Debian:

```bash
sudo apt update
sudo apt install build-essential pkg-config libopencv-dev
```

## Dataset

Loader yêu cầu ảnh `.png`, `.jpg` hoặc `.jpeg`, kích thước `32x32`, theo cấu trúc:

```text
datasets/cifar10/
├── train/
│   ├── airplane/ automobile/ bird/ cat/ deer/
│   ├── dog/ frog/ horse/ ship/ truck/
└── test/
    ├── airplane/ automobile/ bird/ cat/ deer/
    └── dog/ frog/ horse/ ship/ truck/
```

Nhãn được ánh xạ theo thứ tự: `airplane=0`, `automobile=1`, `bird=2`, `cat=3`, `deer=4`, `dog=5`, `frog=6`, `horse=7`, `ship=8`, `truck=9`.

Ảnh được đọc ở dạng channels-last `[32, 32, 3]` và chuẩn hóa pixel về `[0, 1]`.

Kiểm tra loader:

```bash
make test-cifar10
```

## Build và chạy

```bash
make
make train-cifar10
```

Train nhận hai tham số:

```text
./build/train_cifar10 [số_epoch] [số_sample_tối_đa]
```

`0` ở tham số sample nghĩa là dùng toàn bộ split tương ứng. Ví dụ:

```bash
# 3 epoch, tối đa 100 sample train và 100 sample test
./build/train_cifar10 3 100

# 10 epoch trên toàn bộ train/test
./build/train_cifar10 10 0
```

Mặc định:

```text
epochs       = 1
max_samples  = 100
learning_rate = 0.001
```

Sau mỗi epoch chương trình in:

- train loss/accuracy sau khi hoàn tất toàn bộ update của epoch;
- test loss/accuracy;
- số update thành công và sample lỗi;
- thời gian epoch và tốc độ update.

## Luồng huấn luyện

Mỗi sample đi qua toàn bộ pipeline:

```text
image
  -> patch embedding
  -> CLS token + positional embedding
  -> encoder block 1..N
  -> classifier
  -> cross entropy
  -> backward qua classifier, encoder, positional/CLS và patch embedding
  -> cập nhật với batch size = 1
```

Các tham số được cập nhật gồm patch embedding, CLS token, positional embedding, LayerNorm gamma/beta, Q/K/V/output projection, MLP và classification head. CLS token và positional embedding được khởi tạo bằng Xavier thay vì zero cố định. Mỗi encoder block dùng seed riêng.

Do update xảy ra ngay sau từng sample (batch size = 1), vòng train chính hiện tuần tự để bảo đảm thứ tự cập nhật và tránh race condition trên shared parameters. OpenMP vẫn được bật cho build và dùng để đo thời gian; `OMP_NUM_THREADS` không biến cách cập nhật batch size = 1 thành update song song.

Loss dùng để in sau epoch được tính lại bằng một evaluation pass, nên không bị nhầm với loss trước update của từng sample.

## Cấu hình mô hình

| Thành phần | Giá trị |
|---|---:|
| Ảnh | `32x32x3` |
| Patch | `8x8` |
| Số patch | `16` |
| Embedding dimension | `64` |
| Attention heads | `4` |
| MLP hidden dimension | `256` |
| Encoder blocks | `2` |
| Số class | `10` |
| Learning rate | `0.001` |

Mỗi ảnh tạo `16` patch và `1` CLS token, tổng cộng `17` token.

## Các lệnh kiểm tra khác

```bash
# Kiểm tra patch embedding, encoder, logits và full ViT update
make test-patches

# Chạy demo forward với 0010.png
make run

# Xóa binary build
make clean
```

`dowload_datasets.py` là script mẫu dùng Ultralytics và không tự tạo đúng cấu trúc thư mục mà loader C++ yêu cầu.
