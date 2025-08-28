#include <iostream>
#include <opencv2/opencv.hpp>
#include <Windows.h>
#include <chrono>
#include "SharedMemory.h" // 独自定義の共有メモリアクセス用クラス
#include <nvtx3/nvtx3.hpp>

// 共有メモリを監視して更新があったら通知する（ダミー）
void MonitorSharedMemory()
{
    // 何もしない
}

int main()
{
    // 共有メモリを開く
    SharedMemory shm("test", 1920 * 1080 * 3);

    // 共有メモリのデータを読み込むバッファ
    char* data = new char[shm.getSize()];

    // ウィンドウを作成
    cv::namedWindow("img", cv::WINDOW_AUTOSIZE);

    // 共有メモリから画像データを取得して表示
    while (true)
    {
        // 処理時間を計測
        auto start = std::chrono::high_resolution_clock::now();
        nvtxRangePushA("Frame");

        // 共有メモリから画像データを取得
        shm.read(data);

        // 共有メモリの更新を監視
        if (shm.isUpdated())
        {
            // 画像データをデコード
            cv::Mat img = cv::imdecode(cv::Mat(1, shm.getSize(), CV_8UC1, data), cv::IMREAD_COLOR);

            // 画像を表示
            cv::imshow("img", img);

            // ログを出力
            std::cout << "show" << std::endl;
        }

        // 処理時間を計測
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // std::cout << "処理時間: " << elapsed.count() << " ms" << std::endl;

        // 待機
    cv::waitKey(1); // これがないと固まる

        nvtxRangePop();
    }

    // 後処理
    delete[] data;
    cv::destroyAllWindows();

    return 0;
}
