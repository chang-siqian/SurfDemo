#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>

// Adaptive ROI: row-wise disparity projection to find obstacle band
struct AdaptiveROI {
    int yStart;
    int height;
};

/// Debug: print disparity projection for analysis
void printDisparityStats(const cv::Mat& disp) {
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(disp, &minVal, &maxVal, &minLoc, &maxLoc);
    int validCount = 0;
    for (int r = 0; r < disp.rows; r++) {
        const short* d = disp.ptr<short>(r);
        for (int c = 0; c < disp.cols; c++)
            if (d[c] > 0) validCount++;
    }
    double validPct = validCount * 100.0 / disp.total();
    std::cout << "[STATS] Disparity range: " << minVal << " ~ " << maxVal
              << " | Valid pixels: " << validPct << "%" << std::endl;
}

AdaptiveROI computeAdaptiveROI(const cv::Mat& disp, int imgRows,
                               double energyRatio = 0.80,
                               double marginRatio = 0.10) {
    // Adaptive ROI by disparity energy concentration:
    // Find the narrowest contiguous band that contains `energyRatio`
    // of the total row-wise disparity energy.

    // 1. Row-wise disparity sum ("obstacle energy" per row)
    std::vector<double> rowEnergy(imgRows, 0.0);
    double totalEnergy = 0.0;
    for (int r = 0; r < disp.rows; r++) {
        const short* d = disp.ptr<short>(r);
        double sum = 0;
        for (int c = 0; c < disp.cols; c++) {
            if (d[c] > 0) sum += d[c];  // closer = higher disparity = more energy
        }
        rowEnergy[r] = sum;
        totalEnergy += sum;
    }

    if (totalEnergy < 1.0) {
        return { imgRows / 4, imgRows / 2 };  // fallback: no meaningful disparity
    }

    // 2. Sliding window: find minimal-height band with >= energyRatio of total energy
    double target = totalEnergy * energyRatio;
    int bestH = imgRows;
    int bestY = imgRows / 4;
    int top = 0, bot = 0;
    double windowSum = 0;

    while (bot < imgRows) {
        windowSum += rowEnergy[bot++];
        while (top < bot && windowSum - rowEnergy[top] >= target) {
            windowSum -= rowEnergy[top++];
        }
        if (windowSum >= target) {
            int h = bot - top;
            if (h < bestH) {
                bestH = h;
                bestY = top;
            }
        }
    }

    // 3. Add margin
    int margin = static_cast<int>(imgRows * marginRatio);
    int yStart = (std::max)(0, bestY - margin);
    int yEnd   = (std::min)(imgRows - 1, bestY + bestH - 1 + margin);
    int roiH   = yEnd - yStart + 1;

    // Enforce minimum 20% height
    int minH = (std::max)(1, imgRows / 5);
    if (roiH < minH) {
        int center = (yStart + yEnd) / 2;
        yStart = (std::max)(0, center - minH / 2);
        yEnd   = (std::min)(imgRows - 1, yStart + minH - 1);
        roiH   = yEnd - yStart + 1;
    }
    return { yStart, roiH };
}

int main() {
    std::string pathL = "left.jpg";
    std::string pathR = "right.jpg";

    cv::Mat imgL = cv::imread(pathL, cv::IMREAD_GRAYSCALE);
    cv::Mat imgR = cv::imread(pathR, cv::IMREAD_GRAYSCALE);

    if (imgL.empty() || imgR.empty()) {
        std::cout << "[ERROR] Cannot load stereo images" << std::endl;
        return -1;
    }
    if (imgL.size() != imgR.size())
        cv::resize(imgR, imgR, imgL.size());

    int imgRows = imgL.rows, imgCols = imgL.cols;
    std::cout << "Image: " << imgCols << "x" << imgRows
              << " (" << (imgL.total() / 1000) << "k pixels)\n" << std::endl;

    int minDisp = 0, numDisp = 128, blkSize = 3;

    // CUDA warmup
    std::cout << "[WARMUP] Initializing CUDA..." << std::endl;
    {
        cv::cuda::GpuMat dL, dR, dD;
        dL.upload(imgL); dR.upload(imgR);
        auto wm = cv::cuda::createStereoSGM(minDisp, numDisp,
            8 * blkSize * blkSize, 32 * blkSize * blkSize, 1,
            cv::cuda::StereoSGM::MODE_HH4);
        wm->compute(dL, dR, dD);
        cv::Mat discard; dD.download(discard);
    }
    std::cout << "[WARMUP] Done.\n" << std::endl;

    // Full-frame SGM to simulate "previous frame" for adaptive ROI
    cv::Mat dispFull;
    {
        cv::cuda::GpuMat dL, dR, dD;
        dL.upload(imgL); dR.upload(imgR);
        auto cuda = cv::cuda::createStereoSGM(minDisp, numDisp,
            8 * blkSize * blkSize, 32 * blkSize * blkSize, 1,
            cv::cuda::StereoSGM::MODE_HH4);
        cv::TickMeter tm; tm.start();
        cuda->compute(dL, dR, dD);
        tm.stop();
        dD.download(dispFull);
        std::cout << "[INIT] Full-frame CUDA SGM: " << tm.getTimeMilli() << " ms" << std::endl;
    }

    // Print disparity statistics
    printDisparityStats(dispFull);

    // Adaptive ROI: find narrowest band containing 80% of disparity energy
    AdaptiveROI aroi = computeAdaptiveROI(dispFull, imgRows, 0.80, 0.10);
    double roiPct = aroi.height * 100.0 / imgRows;
    std::cout << "[ADAPTIVE] ROI: y=" << aroi.yStart
              << " h=" << aroi.height
              << " (" << std::fixed << std::setprecision(1) << roiPct << "% of frame)" << std::endl;

    // Fixed 50% ROI for comparison
    int fixedY = imgRows / 4, fixedH = imgRows / 2;
    double fixedPct = fixedH * 100.0 / imgRows;

    // Build CUDA SGM once, reuse across benchmarks
    auto makeSGM = [&]() {
        return cv::cuda::createStereoSGM(minDisp, numDisp,
            8 * blkSize * blkSize, 32 * blkSize * blkSize, 1,
            cv::cuda::StereoSGM::MODE_HH4);
    };

    struct Result { std::string label; double timeMs; int roiH; };
    std::vector<Result> results;

    std::cout << "\n--- Benchmark (CUDA SGM) ---" << std::endl;

    // No ROI
    {
        cv::cuda::GpuMat dL, dR, dD;
        dL.upload(imgL); dR.upload(imgR);
        auto cuda = makeSGM();
        cv::TickMeter tm; tm.start();
        cuda->compute(dL, dR, dD);
        tm.stop();
        results.push_back({"No ROI (full frame)", tm.getTimeMilli(), imgRows});
        std::cout << "  No ROI (full)            " << std::setw(8) << tm.getTimeMilli() << " ms" << std::endl;
    }

    // Fixed ROI (50%)
    {
        cv::Rect roi(0, fixedY, imgCols, fixedH);
        cv::Mat lr = imgL(roi), rr = imgR(roi);
        cv::cuda::GpuMat dL, dR, dD;
        dL.upload(lr); dR.upload(rr);
        auto cuda = makeSGM();
        cv::TickMeter tm; tm.start();
        cuda->compute(dL, dR, dD);
        tm.stop();
        results.push_back({"Fixed ROI (50%)", tm.getTimeMilli(), fixedH});
        std::cout << "  Fixed ROI (50%)          " << std::setw(8) << tm.getTimeMilli() << " ms" << std::endl;
    }

    // Adaptive ROI
    {
        cv::Rect roi(0, aroi.yStart, imgCols, aroi.height);
        cv::Mat lr = imgL(roi), rr = imgR(roi);
        cv::cuda::GpuMat dL, dR, dD;
        dL.upload(lr); dR.upload(rr);
        auto cuda = makeSGM();
        cv::TickMeter tm; tm.start();
        cuda->compute(dL, dR, dD);
        tm.stop();
        results.push_back({"Adaptive ROI (" + std::to_string((int)roiPct) + "%)", tm.getTimeMilli(), aroi.height});
        std::cout << "  Adaptive ROI (" << (int)roiPct << "%)        " << std::setw(8) << tm.getTimeMilli() << " ms" << std::endl;
    }

    // Summary table
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << std::left  << std::setw(30) << "Method"
              << std::right << std::setw(12) << "Time(ms)"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "ROI H(px)"
              << std::setw(10) << "ROI%" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    double baseline = results[0].timeMs;
    for (const auto& r : results) {
        double speedup = baseline / r.timeMs;
        double pct = r.roiH * 100.0 / imgRows;
        std::cout << std::left  << std::setw(30) << r.label
                  << std::right << std::setw(10) << std::setprecision(1) << r.timeMs
                  << std::setw(10) << std::setprecision(1) << speedup << "x"
                  << std::setw(10) << r.roiH
                  << std::setw(8) << std::setprecision(1) << pct << "%" << std::endl;
    }
    std::cout << std::string(70, '=') << std::endl;

    // Visualization: overlay ROI boundaries on disparity map
    cv::Mat dispVis;
    dispFull.convertTo(dispVis, CV_8U, 255.0 / (numDisp * 16.0));
    cv::cvtColor(dispVis, dispVis, cv::COLOR_GRAY2BGR);

    cv::line(dispVis, cv::Point(0, aroi.yStart), cv::Point(imgCols-1, aroi.yStart),
             cv::Scalar(0, 255, 0), 2);
    cv::line(dispVis, cv::Point(0, aroi.yStart + aroi.height - 1),
             cv::Point(imgCols-1, aroi.yStart + aroi.height - 1), cv::Scalar(0, 255, 0), 2);
    cv::line(dispVis, cv::Point(0, fixedY), cv::Point(imgCols-1, fixedY),
             cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
    cv::line(dispVis, cv::Point(0, fixedY + fixedH - 1),
             cv::Point(imgCols-1, fixedY + fixedH - 1), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

    cv::putText(dispVis, "Adaptive", cv::Point(5, aroi.yStart - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    cv::putText(dispVis, "Fixed 50%", cv::Point(5, fixedY - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);

    cv::namedWindow("Left Image", cv::WINDOW_NORMAL);
    cv::imshow("Left Image", imgL);
    cv::namedWindow("Adaptive ROI Comparison", cv::WINDOW_NORMAL);
    cv::imshow("Adaptive ROI Comparison", dispVis);
    cv::waitKey(0);
    return 0;
}
