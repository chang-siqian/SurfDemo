#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <iostream>

int main() {
    std::cout << "Step 1: loading..." << std::endl;
    cv::Mat L = cv::imread("x64/Debug/left.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat R = cv::imread("x64/Debug/right.jpg", cv::IMREAD_GRAYSCALE);
    std::cout << "Step 2: loaded " << L.cols << "x" << L.rows << std::endl;

    // CUDA warmup (required by CUDA build's OpenCV)
    std::cout << "Step 2b: CUDA warmup..." << std::endl;
    {
        cv::cuda::GpuMat dL, dR, dD;
        dL.upload(L); dR.upload(R);
        auto wm = cv::cuda::createStereoSGM(0, 128, 72, 288, 1, cv::cuda::StereoSGM::MODE_HH4);
        wm->compute(dL, dR, dD);
    }
    std::cout << "Step 2c: warmup done" << std::endl;

    std::cout << "Step 3: CPU SGBM..." << std::endl;
    auto cpu = cv::StereoSGBM::create(0, 128, 3, 72, 288, 1, 63, 10, 100, 1);
    cv::Mat disp;
    cpu->compute(L, R, disp);
    std::cout << "Step 4: done, raw range check..." << std::endl;

    double minV, maxV;
    cv::minMaxLoc(disp, &minV, &maxV);
    std::cout << "Step 5: raw=[" << minV << "," << maxV << "]" << std::endl;

    int invalid = 0;
    for (int r = 0; r < disp.rows; r++) {
        const short* d = disp.ptr<short>(r);
        for (int c = 0; c < disp.cols; c++)
            if (d[c] <= 0) invalid++;
    }
    std::cout << "Step 6: invalid=" << (invalid*100.0/disp.total()) << "%" << std::endl;

    std::cout << "Step 7: saving..." << std::endl;
    cv::Mat vis;
    disp.convertTo(vis, CV_8U, 255.0 / (128 * 16.0));
    cv::applyColorMap(vis, vis, cv::COLORMAP_JET);
    cv::imwrite("x64/Debug/diag_simple_cpu.png", vis);
    std::cout << "Step 8: Done!" << std::endl;
    return 0;
}
