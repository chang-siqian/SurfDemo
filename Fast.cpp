#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

/*================= 你只需要改这里 =================*/
// 两张图片路径 - 使用绝对路径或确保图片在正确目录
static string PATH1 = "img1.jpg";
static string PATH2 = "img2.jpg";

// FAST 参数（核心）
static int  FAST_THRESHOLD = 40;      // 阈值：越小点越多（常见 5~40）
static bool FAST_NONMAX_SUPPRESS = true;    // 是否做非极大值抑制
static FastFeatureDetector::DetectorType FAST_TYPE =
FastFeatureDetector::TYPE_9_16;          // 也可改为 TYPE_7_12 或 TYPE_5_8

// 均匀化（可选）：半径NMS，控制点的“分散度”
static bool  USE_RADIUS_NMS = true;        // 开启后角点更均匀
static float RADIUS_MIN_DIST = 10.0f;        // 半径像素，越大越分散

// 显示/导出
static int  TARGET_H = 700;         // 统一显示高度
static bool SAVE_OUTPUT = true;        // 是否保存拼接图
/*==================================================*/

static Mat resizeToHeight(const Mat& img, int h) {
    if (img.empty()) return Mat();
    double s = h / (double)img.rows;
    Mat r; resize(img, r, Size(), s, s, INTER_AREA);
    return r;
}

// 半径NMS：按 response 从高到低保留，彼此距离须 >= RADIUS_MIN_DIST
static void radiusNMS(vector<KeyPoint>& kps, float radius) {
    if (kps.empty()) return;
    sort(kps.begin(), kps.end(),
        [](const KeyPoint& a, const KeyPoint& b) { return a.response > b.response; });
    vector<KeyPoint> kept; kept.reserve(kps.size());
    const float r2 = radius * radius;

    for (const auto& kp : kps) {
        bool keep = true;
        for (const auto& sel : kept) {
            float dx = kp.pt.x - sel.pt.x;
            float dy = kp.pt.y - sel.pt.y;
            if (dx * dx + dy * dy < r2) { keep = false; break; }
        }
        if (keep) kept.push_back(kp);
    }
    kps.swap(kept);
}

static Mat detectFASTandDraw(const Mat& src) {
    if (src.empty()) {
        cerr << "[ERROR] 输入图像为空!" << endl;
        return Mat();
    }

    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    }
    else if (src.channels() == 1) {
        gray = src.clone();
    }
    else {
        cerr << "[ERROR] 不支持的图像通道数: " << src.channels() << endl;
        return src.clone();
    }

    if (gray.empty()) {
        cerr << "[ERROR] 灰度转换失败!" << endl;
        return src.clone();
    }

    Ptr<FastFeatureDetector> fd = FastFeatureDetector::create(
        FAST_THRESHOLD, FAST_NONMAX_SUPPRESS, FAST_TYPE
    );
    vector<KeyPoint> kps;
    fd->detect(gray, kps);

    cout << "检测到 " << kps.size() << " 个特征点" << endl;

    if (USE_RADIUS_NMS && !kps.empty()) {
        radiusNMS(kps, RADIUS_MIN_DIST);
    }

    Mat vis = src.clone(); // 先克隆原图
    try {
        drawKeypoints(src, kps, vis, Scalar(0, 0, 255),
            DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        putText(vis, string("#") + to_string(kps.size()), Point(16, 42),
            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2, LINE_AA);
    }
    catch (const Exception& e) {
        cerr << "[EXCEPTION] drawKeypoints 错误: " << e.what() << endl;
        // 保持原图不变
    }

    return vis;
}

int main() {
    cout << "OpenCV FAST 特征检测器" << endl;
    cout << "图片1路径: " << PATH1 << endl;
    cout << "图片2路径: " << PATH2 << endl;

    Mat img1 = imread(PATH1), img2 = imread(PATH2);
    if (img1.empty() || img2.empty()) {
        cerr << "[ERROR] 无法读取图片，请检查 PATH1/2。" << endl;
        cerr << "PATH1: " << PATH1 << " - " << (img1.empty() ? "空" : "成功") << endl;
        cerr << "PATH2: " << PATH2 << " - " << (img2.empty() ? "空" : "成功") << endl;

        // 尝试使用示例图片
        cout << "尝试创建示例图片..." << endl;
        img1 = Mat::zeros(500, 500, CV_8UC3);
        img2 = Mat::zeros(500, 500, CV_8UC3);
        rectangle(img1, Point(100, 100), Point(400, 400), Scalar(255, 0, 0), -1);
        rectangle(img2, Point(150, 150), Point(350, 350), Scalar(0, 255, 0), -1);
    }
    else {
        cout << "图像1尺寸: " << img1.cols << "x" << img1.rows << endl;
        cout << "图像2尺寸: " << img2.cols << "x" << img2.rows << endl;
    }

    Mat res1 = detectFASTandDraw(img1);
    Mat res2 = detectFASTandDraw(img2);

    if (res1.empty() || res2.empty()) {
        cerr << "[ERROR] 特征检测失败!" << endl;
        return -1;
    }

    res1 = resizeToHeight(res1, TARGET_H);
    res2 = resizeToHeight(res2, TARGET_H);

    Mat combined;
    hconcat(res1, res2, combined);

    putText(combined, "Image 1", Point(30, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
    putText(combined, "Image 2", Point(res1.cols + 30, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

    int maxW = 1400;
    if (combined.cols > maxW) {
        double s = maxW / (double)combined.cols;
        resize(combined, combined, Size(), s, s, INTER_AREA);
    }

    namedWindow("FAST Comparison", WINDOW_NORMAL);
    resizeWindow("FAST Comparison", combined.cols, combined.rows);
    imshow("FAST Comparison", combined);

    if (SAVE_OUTPUT) {
        imwrite("fast_comparison.jpg", combined);
        cout << "结果已保存为: fast_comparison.jpg" << endl;
    }

    waitKey(0);
    return 0;
}