#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

int main() {
    // ===== 1. 加载立体图像 =====
    std::string pathL = "left.jpg";
    std::string pathR = "right.jpg";

    cv::Mat imgL = cv::imread(pathL, cv::IMREAD_GRAYSCALE);
    cv::Mat imgR = cv::imread(pathR, cv::IMREAD_GRAYSCALE);

    if (imgL.empty() || imgR.empty()) {
        std::cout << "[ERROR] Cannot load stereo images" << std::endl;
        return -1;
    }
    if (imgL.size() != imgR.size()) {
        cv::resize(imgR, imgR, imgL.size());
    }

    std::cout << "Image: " << imgL.cols << "x" << imgL.rows << " (" << (imgL.total()/1000) << "k pixels)" << std::endl;

    // ===== 2. 参数 =====
    int minDisp = 0;
    int numDisp = 128;
    int blkSize = 3;

    // ===== 3. CUDA 预热 (消除首次编译/初始化开销) =====
    std::cout << "\n[WARMUP] Initializing CUDA..." << std::endl;
    {
        cv::cuda::GpuMat dL, dR, dD;
        dL.upload(imgL);
        dR.upload(imgR);
        cv::Ptr<cv::cuda::StereoSGM> wm = cv::cuda::createStereoSGM(
            minDisp, numDisp, 8*blkSize*blkSize, 32*blkSize*blkSize, 1,
            cv::cuda::StereoSGM::MODE_HH4);
        wm->compute(dL, dR, dD);
        cv::Mat discard;
        dD.download(discard);
    }
    std::cout << "[WARMUP] Done.\n" << std::endl;

    // ===== 4. 定义实验 =====
    struct Test {
        std::string label;
        bool        useCuda;
        bool        useRoi;
    };
    std::vector<Test> tests = {
        {"CPU SGBM (full)",    false, false},
        {"CPU SGBM (ROI)",     false, true},
        {"CUDA SGM  (full)",   true,  false},
        {"CUDA SGM  (ROI)",    true,  true},
    };

    struct Result {
        std::string label;
        double      timeMs;
        cv::Mat     disp;
    };
    std::vector<Result> results;

    // ROI 定义
    int yStart = imgL.rows / 4;
    int roiH   = imgL.rows / 2;
    cv::Rect roi(0, yStart, imgL.cols, roiH);
    cv::Mat imgL_roi = imgL(roi);
    cv::Mat imgR_roi = imgR(roi);

    for (const auto& t : tests) {
        cv::Mat left  = t.useRoi ? imgL_roi : imgL;
        cv::Mat right = t.useRoi ? imgR_roi : imgR;
        cv::Mat disp;

        cv::TickMeter tm;
        tm.start();

        if (t.useCuda) {
            cv::cuda::GpuMat dL, dR, dD;
            dL.upload(left);
            dR.upload(right);

            cv::Ptr<cv::cuda::StereoSGM> cuda = cv::cuda::createStereoSGM(
                minDisp, numDisp,
                8 * blkSize * blkSize,
                32 * blkSize * blkSize,
                1, cv::cuda::StereoSGM::MODE_HH4);
            cuda->compute(dL, dR, dD);
            dD.download(disp);
        } else {
            cv::Ptr<cv::StereoSGBM> cpu = cv::StereoSGBM::create(
                minDisp, numDisp, blkSize,
                8 * blkSize * blkSize,
                32 * blkSize * blkSize,
                1, 63, 10, 100, 1);
            cpu->compute(left, right, disp);
        }

        tm.stop();
        results.push_back({t.label, tm.getTimeMilli(), disp.clone()});
        std::cout << "  " << std::left << std::setw(22) << t.label
                  << std::right << std::setw(8) << std::fixed << std::setprecision(1)
                  << tm.getTimeMilli() << " ms" << std::endl;
    }

    // ===== 5. 汇总 =====
    std::cout << "\n" << std::string(65, '=') << std::endl;
    std::cout << std::left << std::setw(28) << "Method"
              << std::right << std::setw(12) << "Time(ms)"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "Pixels" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    double baseline = results[0].timeMs;
    for (const auto& r : results) {
        double speedup = baseline / r.timeMs;
        int px = r.disp.total();
        std::cout << std::left << std::setw(28) << r.label
                  << std::right << std::setw(10) << std::setprecision(1) << r.timeMs
                  << std::setw(10) << std::setprecision(1) << speedup << "x"
                  << std::setw(8) << (px/1000) << "k" << std::endl;
    }
    std::cout << std::string(65, '=') << std::endl;

    // ===== 6. 显示 =====
    for (const auto& r : results) {
        cv::Mat vis;
        r.disp.convertTo(vis, CV_8U, 255.0 / (numDisp * 16.0));
        cv::namedWindow(r.label, cv::WINDOW_NORMAL);
        cv::imshow(r.label, vis);
    }
    cv::waitKey(0);
    return 0;
}
