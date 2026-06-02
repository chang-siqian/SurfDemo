#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <filesystem>
using namespace cv;
using namespace cv::xfeatures2d;

int main() {
    Mat img1 = imread("img1.jpg", IMREAD_GRAYSCALE);
    Mat img2 = imread("img2.jpg", IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        std::cerr << "❌ 无法加载图像！请把两张图片放在可执行文件所在目录。\n";
        return -1;
    }

    // 创建 SIFT 检测器（需要 NONFREE 已启用）
    cv::Ptr<cv::Feature2D> f2d = cv::SIFT::create(
        /*nfeatures=*/0, /*nOctaveLayers=*/3,
        /*contrastThreshold=*/0.04, /*edgeThreshold=*/10, /*sigma=*/1.6
    );
    std::vector<KeyPoint> key1, key2;
    Mat des1, des2;
    cv::TickMeter t_det1, t_desc1, t_det2, t_desc2;

    // 图1：先 detect 再 compute（分别计时）
    t_det1.start();
    f2d->detect(img1, key1);
    t_det1.stop();

    t_desc1.start();
    f2d->compute(img1, key1, des1);
    t_desc1.stop();

    // 图2：同样分开计时
    t_det2.start();
    f2d->detect(img2, key2);
    t_det2.stop();

    t_desc2.start();
    f2d->compute(img2, key2, des2);
    t_desc2.stop();

    // 打印关键点数与时间
    std::cout << "[SIFT] #kp1=" << key1.size() << "  #kp2=" << key2.size() << "\n";
    std::cout << "[SIFT] detect(ms): img1=" << t_det1.getTimeMilli()
        << "  img2=" << t_det2.getTimeMilli() << "\n";
    std::cout << "[SIFT] describe(ms): img1=" << t_desc1.getTimeMilli()
        << "  img2=" << t_desc2.getTimeMilli() << "\n";

    std::cout << "关键点数: " << key1.size() << " / " << key2.size() << std::endl;

    // === C 点：匹配与 ratio test（计时版）===
    cv::TickMeter t_match, t_ratio;

    // 匹配计时
    t_match.start();
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(des1, des2, knn, 2);
    t_match.stop();

    // ratio test 计时
    t_ratio.start();
    std::vector<cv::DMatch> good;
    good.reserve(knn.size());
    for (auto& m : knn) {
        if (m.size() == 2 && m[0].distance < 0.75f * m[1].distance)
            good.push_back(m[0]);
    }
    t_ratio.stop();
    std::cout << "[SIFT] match(ms)=" << t_match.getTimeMilli()
        << "  ratio(ms)=" << t_ratio.getTimeMilli()
        << "  #good=" << good.size() << std::endl;


    // === D 点：RANSAC 估计内点率 & 重投影误差 ===
    double inlier_ratio = 0.0;
    double reproj_err = 0.0;
    int    inliers = 0;

    if (good.size() >= 4) {
        std::vector<cv::Point2f> p1, p2;
        p1.reserve(good.size()); p2.reserve(good.size());
        for (const auto& m : good) {
            p1.push_back(key1[m.queryIdx].pt);
            p2.push_back(key2[m.trainIdx].pt);
        }

        std::vector<unsigned char> inlierMask;
        cv::Mat H = cv::findHomography(p1, p2, cv::RANSAC, 3.0, inlierMask); // 3px 阈值

        inliers = std::accumulate(inlierMask.begin(), inlierMask.end(), 0);
        inlier_ratio = inliers / std::max<int>(1, static_cast<int>(good.size()));

        if (!H.empty() && inliers > 0) {
            std::vector<cv::Point2f> p1_proj;
            cv::perspectiveTransform(p1, p1_proj, H);
            double sum_err = 0.0;
            int cnt = 0;
            for (size_t i = 0; i < p1.size(); ++i) if (inlierMask[i]) {
                double dx = p1_proj[i].x - p2[i].x;
                double dy = p1_proj[i].y - p2[i].y;
                sum_err += std::sqrt(dx * dx + dy * dy);
                cnt++;
            }
            reproj_err = (cnt > 0) ? (sum_err / cnt) : 0.0;
        }

        std::cout << "[SIFT] inliers=" << inliers
            << "  inlier_ratio=" << inlier_ratio
            << "  reproj_err(px)=" << reproj_err << std::endl;
    }
    else {
        std::cout << "[SIFT] Not enough matches for RANSAC" << std::endl;
    }

    // === E 点：描述子内存估算（更通用，自动依据 Mat 类型/列数计算）===
    size_t bytes_per_desc = static_cast<size_t>(des1.cols) * des1.elemSize(); // 每个描述子的字节数
    size_t mem_img1 = static_cast<size_t>(des1.rows) * bytes_per_desc;
    size_t mem_img2 = static_cast<size_t>(des2.rows) * bytes_per_desc;

    auto toMB = [](size_t b) { return b / (1024.0 * 1024.0); };
    std::cout << "[SIFT] descriptor_len=" << bytes_per_desc << " bytes"
        << "  mem(img1)=" << toMB(mem_img1) << " MB"
        << "  mem(img2)=" << toMB(mem_img2) << " MB" << std::endl;

    // =========================
    // 绘制匹配结果（改进版）
    // =========================

    // 1) 只保留前 N 个匹配（按距离从小到大）
    std::sort(good.begin(), good.end(),
        [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });
    int N = std::min<int>(60, static_cast<int>(good.size()));
    std::vector<cv::DMatch> good_top(good.begin(), good.begin() + N);

    // 2) 画拼接+连线（不画孤立关键点）
    cv::Mat vis;
    cv::drawMatches(img1, key1, img2, key2, good_top, vis,
        cv::Scalar::all(-1), cv::Scalar::all(-1),
        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // 3) 窗口可缩放
    cv::namedWindow("SIFT Matches", cv::WINDOW_NORMAL);

    // 4) 宽图自动缩放到不超过 1200px，并保存到文件
    int targetW = 1200;
    if (vis.cols > targetW) {
        double scale = static_cast<double>(targetW) / vis.cols;
        cv::Mat vis_small;
        cv::resize(vis, vis_small, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::imshow("SIFT Matches", vis_small);
        cv::imwrite("matches.jpg", vis_small);
    }
    else {
        cv::imshow("SIFT Matches", vis);
        cv::imwrite("matches.jpg", vis);
    }

    // 5) 等待按键退出
    cv::waitKey(0);

}
