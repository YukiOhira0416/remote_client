#pragma once
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <cstring> // For std::memcpy
#include <iostream>

struct BmpTask {
    std::string filename;
    int width{0};
    int height{0};
    size_t pitch{0};          // 8bppの行ピッチ
    std::vector<uint8_t> data; // 高さ×pitch
};

class BmpWriter {
public:
    BmpWriter()
        : stop_(false), th_([this]{ run(); }) {}

    ~BmpWriter() {
        {
            std::lock_guard<std::mutex> lk(mx_);
            stop_ = true;
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }

    void enqueue(BmpTask&& t) {
        {
            std::lock_guard<std::mutex> lk(mx_);
            q_.push(std::move(t));
        }
        cv_.notify_one();
    }

private:
    void run() {
        for (;;) {
            BmpTask t;
            {
                std::unique_lock<std::mutex> lk(mx_);
                cv_.wait(lk, [&]{ return stop_ || !q_.empty(); });
                if (stop_ && q_.empty()) break;
                t = std::move(q_.front());
                q_.pop();
            }
            (void)save8u(t.filename, t.data.data(), t.width, t.height, t.pitch);
        }
    }

    static bool save8u(const std::string& filename,
                       const uint8_t* src, int width, int height, size_t srcPitch) {
#pragma pack(push, 1)
        struct BmpFileHeader {
            uint16_t bfType{0x4D42};
            uint32_t bfSize{0};
            uint16_t bfReserved1{0};
            uint16_t bfReserved2{0};
            uint32_t bfOffBits{0};
        };
        struct BmpInfoHeader {
            uint32_t biSize{40};
            int32_t  biWidth{0};
            int32_t  biHeight{0};
            uint16_t biPlanes{1};
            uint16_t biBitCount{8};
            uint32_t biCompression{0};
            uint32_t biSizeImage{0};
            int32_t  biXPelsPerMeter{2835};
            int32_t  biYPelsPerMeter{2835};
            uint32_t biClrUsed{256};
            uint32_t biClrImportant{0};
        };
#pragma pack(pop)

        if (!src || width <= 0 || height <= 0) return false;

        const uint32_t rowBytes = (uint32_t)((width + 3) & ~3u);
        const uint32_t pixelsSize = rowBytes * (uint32_t)height;
        const uint32_t paletteSize = 256 * 4;
        const uint32_t headerSize = sizeof(BmpFileHeader) + sizeof(BmpInfoHeader) + paletteSize;

        BmpFileHeader fh;
        BmpInfoHeader ih;
        ih.biWidth = width;
        ih.biHeight = height; // Bottom-up format
        ih.biSizeImage = pixelsSize;
        fh.bfOffBits = headerSize;
        fh.bfSize = headerSize + pixelsSize;

        const std::string tmp = filename + ".tmp";
        std::ofstream ofs(tmp, std::ios::binary);
        if (!ofs) return false;

        ofs.write(reinterpret_cast<const char*>(&fh), sizeof(fh));
        ofs.write(reinterpret_cast<const char*>(&ih), sizeof(ih));
        for (int i = 0; i < 256; ++i) {
            const unsigned char bgra[4] = { (unsigned char)i, (unsigned char)i, (unsigned char)i, 0 };
            ofs.write(reinterpret_cast<const char*>(bgra), 4);
        }
        std::vector<uint8_t> row(rowBytes, 0);
        for (int y = height - 1; y >= 0; --y) {
            const uint8_t* srcLine = src + (size_t)y * srcPitch;
            std::memcpy(row.data(), srcLine, (size_t)width);
            ofs.write(reinterpret_cast<const char*>(row.data()), rowBytes);
        }
        ofs.flush();
        ofs.close();

        std::error_code ec;
        std::filesystem::remove(filename, ec);
        std::filesystem::rename(tmp, filename, ec);
        if (ec) {
            std::filesystem::copy_file(tmp, filename,
                std::filesystem::copy_options::overwrite_existing, ec);
            std::filesystem::remove(tmp, ec);
        }
        return true;
    }

    std::mutex mx_;
    std::condition_variable cv_;
    std::queue<BmpTask> q_;
    bool stop_;
    std::thread th_;
};
