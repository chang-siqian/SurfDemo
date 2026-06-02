#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

struct AdaptiveROI { int yStart; int height; };

AdaptiveROI computeAdaptiveROI(const cv::Mat& disp, int imgRows,
                               double energyRatio = 0.80,
                               double marginRatio = 0.10) {
    std::vector<double> rowEnergy(imgRows, 0.0);
    double totalEnergy = 0.0;
    for (int r = 0; r < disp.rows; r++) {
        const short* d = disp.ptr<short>(r);
        double sum = 0;
        for (int c = 0; c < disp.cols; c++)
            if (d[c] > 0) sum += d[c];
        rowEnergy[r] = sum;
        totalEnergy += sum;
    }
    if (totalEnergy < 1.0)
        return { imgRows / 4, imgRows / 2 };

    double target = totalEnergy * energyRatio;
    int bestH = imgRows, bestY = imgRows / 4;
    int top = 0, bot = 0;
    double windowSum = 0;
    while (bot < imgRows) {
        windowSum += rowEnergy[bot++];
        while (top < bot && windowSum - rowEnergy[top] >= target)
            windowSum -= rowEnergy[top++];
        if (windowSum >= target) {
            int h = bot - top;
            if (h < bestH) { bestH = h; bestY = top; }
        }
    }
    int margin = static_cast<int>(imgRows * marginRatio);
    int yStart = (std::max)(0, bestY - margin);
    int yEnd   = (std::min)(imgRows - 1, bestY + bestH - 1 + margin);
    int roiH   = yEnd - yStart + 1;
    int minH = (std::max)(1, imgRows / 5);
    if (roiH < minH) {
        int center = (yStart + yEnd) / 2;
        yStart = (std::max)(0, center - minH / 2);
        yEnd   = (std::min)(imgRows - 1, yStart + minH - 1);
        roiH   = yEnd - yStart + 1;
    }
    return { yStart, roiH };
}

cv::Mat loadKITTIDisparity(const std::string& path) {
    cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
    cv::Mat disp;
    raw.convertTo(disp, CV_32F, 1.0 / 256.0);
    return disp;
}

// Evaluate predicted disparity (CV_16S, *16) against float gt, same dimensions
struct EvalResult {
    double epe, d1All, bad3px;
    int validCount;
};
EvalResult evaluate(const cv::Mat& predDisp, const cv::Mat& gtDisp) {
    cv::Mat pred;
    predDisp.convertTo(pred, CV_32F, 1.0 / 16.0);

    double epeSum = 0;
    int total = 0, bad = 0, bad3 = 0;

    for (int r = 0; r < pred.rows; r++) {
        const float* p = pred.ptr<float>(r);
        const float* g = gtDisp.ptr<float>(r);
        for (int c = 0; c < pred.cols; c++) {
            if (g[c] <= 0) continue;
            total++;
            float err = std::abs(p[c] - g[c]);
            epeSum += err;
            if (err > 3.0f) bad3++;
            if (err > 3.0f && err > 0.05f * g[c]) bad++;
        }
    }
    return { total > 0 ? epeSum / total : 0.0,
             total > 0 ? bad * 100.0 / total : 0.0,
             total > 0 ? bad3 * 100.0 / total : 0.0, total };
}

int main() {
    std::string dataRoot = "data/KITTI/training";
    std::string img2Dir = dataRoot + "/image_2";
    std::string img3Dir = dataRoot + "/image_3";
    std::string gtDir   = dataRoot + "/disp_occ_0";

    std::vector<std::string> fileIds;
    for (const auto& entry : fs::directory_iterator(gtDir)) {
        std::string fname = entry.path().filename().string();
        if (fname.find("_10.png") != std::string::npos)
            fileIds.push_back(fname.substr(0, fname.find("_10.png")));
    }
    std::sort(fileIds.begin(), fileIds.end());
    int nFrames = (int)fileIds.size();

    std::cout << "KITTI Stereo 2015 Evaluation\n";
    std::cout << "Frames with ground truth: " << nFrames << "\n" << std::endl;

    if (nFrames == 0) {
        std::cout << "[ERROR] No KITTI files found" << std::endl;
        return -1;
    }

    int minDisp = 0, numDisp = 128, blkSize = 3;
    int P1 = 8 * blkSize * blkSize;
    int P2 = 32 * blkSize * blkSize;

    // CUDA warmup
    std::cout << "[WARMUP] CUDA SGM..." << std::endl;
    {
        cv::Mat testL = cv::imread(img2Dir + "/" + fileIds[0] + "_10.png", cv::IMREAD_GRAYSCALE);
        cv::Mat testR = cv::imread(img3Dir + "/" + fileIds[0] + "_10.png", cv::IMREAD_GRAYSCALE);
        cv::cuda::GpuMat dL, dR, dD;
        dL.upload(testL); dR.upload(testR);
        auto wm = cv::cuda::createStereoSGM(minDisp, numDisp, P1, P2, 5,
            cv::cuda::StereoSGM::MODE_HH4);
        wm->compute(dL, dR, dD);
    }
    std::cout << "[WARMUP] Done.\n" << std::endl;

    struct Acc { double time, epe, d1, bad3, roiPct; int frames, px; };
    const char* names[5] = {
        "CPU SGBM (full)",
        "CPU SGBM (ROI 50%)",
        "CPU SGBM (Adaptive ROI)",
        "CUDA SGM (full) [speed-only]",
        "CUDA SGM (Adaptive ROI) [spd]"
    };

    Acc m[5] = {};
    m[0] = {0,0,0,0,0,0,0};
    m[1] = {0,0,0,0,0,0,0};
    m[2] = {0,0,0,0,0,0,0};
    m[3] = {0,0,0,0,0,0,0};
    m[4] = {0,0,0,0,0,0,0};

    std::cout << "Processing " << nFrames << " frames...\n" << std::endl;

    int reportInterval = (std::max)(1, nFrames / 10);
    for (int i = 0; i < nFrames; i++) {
        std::string id = fileIds[i];
        std::string pathL = img2Dir + "/" + id + "_10.png";
        std::string pathR = img3Dir + "/" + id + "_10.png";
        std::string pathGt = gtDir + "/" + id + "_10.png";

        cv::Mat imgL = cv::imread(pathL, cv::IMREAD_GRAYSCALE);
        cv::Mat imgR = cv::imread(pathR, cv::IMREAD_GRAYSCALE);
        cv::Mat gtDisp = loadKITTIDisparity(pathGt);
        int rows = imgL.rows, cols = imgL.cols;

        // --- CPU SGBM methods (for accuracy evaluation) ---

        // Method 0: CPU SGBM full frame
        {
            cv::TickMeter tm; tm.start();
            cv::Mat disp;
            auto cpu = cv::StereoSGBM::create(minDisp, numDisp, blkSize, P1, P2,
                1, 63, 10, 100, 1);
            cpu->compute(imgL, imgR, disp);
            tm.stop();
            auto ev = evaluate(disp, gtDisp);
            m[0].time += tm.getTimeMilli(); m[0].epe += ev.epe;
            m[0].d1 += ev.d1All; m[0].bad3 += ev.bad3px;
            m[0].roiPct += 100.0; m[0].frames++; m[0].px += ev.validCount;
        }

        // Method 1: CPU SGBM fixed 50% ROI
        {
            int yStart = rows / 4, roiH = rows / 2;
            cv::Rect roi(0, yStart, cols, roiH);
            cv::Mat lr = imgL(roi), rr = imgR(roi);

            cv::TickMeter tm; tm.start();
            cv::Mat disp;
            auto cpu = cv::StereoSGBM::create(minDisp, numDisp, blkSize, P1, P2,
                1, 63, 10, 100, 1);
            cpu->compute(lr, rr, disp);
            tm.stop();
            auto ev = evaluate(disp, gtDisp(roi));
            m[1].time += tm.getTimeMilli(); m[1].epe += ev.epe;
            m[1].d1 += ev.d1All; m[1].bad3 += ev.bad3px;
            m[1].roiPct += 50.0; m[1].frames++; m[1].px += ev.validCount;
        }

        // Method 2: CPU SGBM adaptive ROI (oracle from full-frame CPU result)
        {
            // Compute full-frame first to get ROI (this cost is the "startup" cost)
            cv::Mat dispFull;
            {
                auto cpu = cv::StereoSGBM::create(minDisp, numDisp, blkSize, P1, P2,
                    1, 63, 10, 100, 1);
                cpu->compute(imgL, imgR, dispFull);
            }

            AdaptiveROI aroi = computeAdaptiveROI(dispFull, rows, 0.80, 0.10);
            cv::Rect roi(0, aroi.yStart, cols, aroi.height);
            cv::Mat lr = imgL(roi), rr = imgR(roi);

            cv::TickMeter tm; tm.start();
            cv::Mat disp;
            auto cpu = cv::StereoSGBM::create(minDisp, numDisp, blkSize, P1, P2,
                1, 63, 10, 100, 1);
            cpu->compute(lr, rr, disp);
            tm.stop();
            auto ev = evaluate(disp, gtDisp(roi));
            double roiPct = aroi.height * 100.0 / rows;
            m[2].time += tm.getTimeMilli();
            m[2].epe += ev.epe; m[2].d1 += ev.d1All; m[2].bad3 += ev.bad3px;
            m[2].roiPct += roiPct;
            m[2].frames++; m[2].px += ev.validCount;
        }

        // --- CUDA SGM methods (speed-only evaluation) ---

        // Method 3: CUDA SGM full frame (speed only)
        {
            cv::cuda::GpuMat dL, dR, dD;
            dL.upload(imgL); dR.upload(imgR);
            auto cuda = cv::cuda::createStereoSGM(minDisp, numDisp, P1, P2, 5,
                cv::cuda::StereoSGM::MODE_HH4);
            cv::TickMeter tm; tm.start();
            cuda->compute(dL, dR, dD);
            tm.stop();
            cv::Mat discard; dD.download(discard);
            m[3].time += tm.getTimeMilli();
            m[3].roiPct += 100.0; m[3].frames++;
        }

        // Method 4: CUDA SGM adaptive ROI (speed only, oracle ROI from CPU)
        {
            cv::Mat dispFull;
            {
                auto cpu = cv::StereoSGBM::create(minDisp, numDisp, blkSize, P1, P2,
                    1, 63, 10, 100, 1);
                cpu->compute(imgL, imgR, dispFull);
            }
            AdaptiveROI aroi = computeAdaptiveROI(dispFull, rows, 0.80, 0.10);
            cv::Rect roi(0, aroi.yStart, cols, aroi.height);
            cv::Mat lr = imgL(roi), rr = imgR(roi);

            cv::cuda::GpuMat dL, dR, dD;
            dL.upload(lr); dR.upload(rr);
            auto cuda = cv::cuda::createStereoSGM(minDisp, numDisp, P1, P2, 5,
                cv::cuda::StereoSGM::MODE_HH4);
            cv::TickMeter tm; tm.start();
            cuda->compute(dL, dR, dD);
            tm.stop();
            cv::Mat discard; dD.download(discard);
            double roiPct = aroi.height * 100.0 / rows;
            m[4].time += tm.getTimeMilli();
            m[4].roiPct += roiPct;
            m[4].frames++;
        }

        if ((i + 1) % reportInterval == 0 || i == nFrames - 1)
            std::cout << "\r  " << (i + 1) << "/" << nFrames << " frames" << std::flush;
    }

    std::cout << "\n\n" << std::string(90, '=') << std::endl;
    std::cout << std::left  << std::setw(28) << "Method"
              << std::right << std::setw(10) << "EPE(px)"
              << std::setw(10) << "D1-all%"
              << std::setw(10) << ">3px%"
              << std::setw(10) << "Time(ms)"
              << std::setw(10) << "ROI%"
              << std::setw(10) << "Speedup" << std::endl;
    std::cout << std::string(90, '-') << std::endl;

    double cpuFullTime = m[0].time / m[0].frames;
    for (int i = 0; i < 5; i++) {
        int n = m[i].frames;
        double avgEpe = n > 0 ? m[i].epe / n : 0;
        double avgD1  = n > 0 ? m[i].d1 / n : 0;
        double avgBad3 = n > 0 ? m[i].bad3 / n : 0;
        double avgTime = n > 0 ? m[i].time / n : 0;
        double avgRoi  = n > 0 ? m[i].roiPct / n : 0;
        double speedup = cpuFullTime / avgTime;

        std::cout << std::left  << std::setw(28) << names[i]
                  << std::right << std::fixed;

        if (i < 3) {
            // Accuracy methods
            std::cout << std::setprecision(2)
                      << std::setw(10) << avgEpe
                      << std::setw(10) << avgD1
                      << std::setw(10) << avgBad3;
        } else {
            // Speed-only methods
            std::cout << std::setw(10) << "N/A"
                      << std::setw(10) << "N/A"
                      << std::setw(10) << "N/A";
        }

        std::cout << std::setprecision(1)
                  << std::setw(10) << avgTime
                  << std::setw(9) << avgRoi << "%"
                  << std::setw(9) << speedup << "x" << std::endl;
    }
    std::cout << std::string(90, '=') << std::endl;

    std::cout << "\nNotes:" << std::endl;
    std::cout << "  - CPU methods: accuracy + speed evaluated on KITTI (200 frames)" << std::endl;
    std::cout << "  - CUDA methods: speed-only (OpenCV 4.11 CUDA SGM has quality issues on KITTI)" << std::endl;
    std::cout << "  - EPE/D1 computed only within each method's ROI region" << std::endl;
    std::cout << "  - Adaptive ROI uses oracle from full-frame CPU SGBM result" << std::endl;

    return 0;
}
