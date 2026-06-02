#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <iostream>

void saveDisparityVis(const cv::Mat& dispRaw, const std::string& filename) {
    cv::Mat vis;
    // Convert CV_16S disparity (*16) to visible image
    dispRaw.convertTo(vis, CV_8U, 255.0 / (128 * 16.0));
    cv::applyColorMap(vis, vis, cv::COLORMAP_JET);

    // Mark invalid pixels (<= 0) in black
    cv::Mat mask;
    cv::threshold(dispRaw, mask, 0, 255, cv::THRESH_BINARY_INV);
    mask.convertTo(mask, CV_8U);
    vis.setTo(cv::Scalar(0, 0, 0), mask);

    cv::imwrite(filename, vis);
    std::cout << "  Saved: " << filename << std::endl;
}

void stats(const std::string& label, const cv::Mat& dispRaw) {
    double minV, maxV;
    cv::minMaxLoc(dispRaw, &minV, &maxV);
    int total = (int)dispRaw.total();
    int invalid = 0;
    for (int r = 0; r < dispRaw.rows; r++) {
        const short* d = dispRaw.ptr<short>(r);
        for (int c = 0; c < dispRaw.cols; c++)
            if (d[c] <= 0) invalid++;
    }
    double invalidPct = invalid * 100.0 / total;
    std::cout << label << ": raw=[" << minV << ", " << maxV
              << "]  invalid=" << invalidPct << "%" << std::endl;
}

int main() {
    int P1 = 72, P2 = 288;

    // ===== Test pair (left.jpg / right.jpg) =====
    std::cout << "\n=== Test pair (left.jpg / right.jpg) ===" << std::endl;
    cv::Mat testL = cv::imread("x64/Debug/left.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat testR = cv::imread("x64/Debug/right.jpg", cv::IMREAD_GRAYSCALE);
    if (testL.size() != testR.size()) cv::resize(testR, testR, testL.size());
    std::cout << "Resolution: " << testL.cols << "x" << testL.rows << std::endl;

    // CUDA warmup (required for CUDA OpenCV build)
    {
        cv::cuda::GpuMat dL, dR, dD;
        dL.upload(testL); dR.upload(testR);
        auto wm = cv::cuda::createStereoSGM(0, 128, 72, 288, 1, cv::cuda::StereoSGM::MODE_HH4);
        wm->compute(dL, dR, dD);
    }
    std::cout << "[WARMUP] Done." << std::endl;

    // CPU SGBM
    {
        cv::Mat disp;
        auto cpu = cv::StereoSGBM::create(0, 128, 3, P1, P2, 1, 63, 10, 100, 1);
        cpu->compute(testL, testR, disp);
        stats("CPU SGBM         ", disp);
        saveDisparityVis(disp, "x64/Debug/diag_test_cpu.png");
    }

    // CUDA SGM
    {
        cv::cuda::GpuMat dL, dR, dD;
        dL.upload(testL); dR.upload(testR);
        auto cuda = cv::cuda::createStereoSGM(0, 128, P1, P2, 1,
            cv::cuda::StereoSGM::MODE_HH4);
        cuda->compute(dL, dR, dD);
        cv::Mat disp; dD.download(disp);
        stats("CUDA SGM (HH4)   ", disp);
        saveDisparityVis(disp, "x64/Debug/diag_test_cuda.png");
    }

    // ===== KITTI pair =====
    std::cout << "\n=== KITTI pair (000000_10) ===" << std::endl;
    cv::Mat kL = cv::imread("data/KITTI/training/image_2/000000_10.png", cv::IMREAD_GRAYSCALE);
    cv::Mat kR = cv::imread("data/KITTI/training/image_3/000000_10.png", cv::IMREAD_GRAYSCALE);
    std::cout << "Resolution: " << kL.cols << "x" << kL.rows << std::endl;

    // CPU SGBM
    {
        cv::Mat disp;
        auto cpu = cv::StereoSGBM::create(0, 128, 3, P1, P2, 1, 63, 10, 100, 1);
        cpu->compute(kL, kR, disp);
        stats("CPU SGBM         ", disp);
        saveDisparityVis(disp, "x64/Debug/diag_kitti_cpu.png");
    }

    // CUDA SGM
    {
        cv::cuda::GpuMat dL, dR, dD;
        dL.upload(kL); dR.upload(kR);
        auto cuda = cv::cuda::createStereoSGM(0, 128, P1, P2, 1,
            cv::cuda::StereoSGM::MODE_HH4);
        cuda->compute(dL, dR, dD);
        cv::Mat disp; dD.download(disp);
        stats("CUDA SGM (HH4)   ", disp);
        saveDisparityVis(disp, "x64/Debug/diag_kitti_cuda.png");
    }

    // ===== Left image for reference =====
    cv::imwrite("x64/Debug/diag_test_left.jpg", testL);
    cv::imwrite("x64/Debug/diag_kitti_left.png", kL);

    std::cout << "\nDone! Check x64/Debug/diag_*.png for visual comparison." << std::endl;
    std::cout << "Black areas = invalid pixels (no match found)" << std::endl;

    return 0;
}
