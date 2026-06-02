#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

/*——— 你只需要改下面这几个“可调参数” ———*/
// 图片路径（改成你的两张图）
static string PATH1 = "img1.jpg";
static string PATH2 = "img2.jpg";

// 角点参数（默认已经偏“多点”）
static int    MAX_CORNERS = 1500;   // 上限：想要更多点就加到 2000/3000
static double QUALITY_LEVEL = 0.008;  // 质量阈值：越小点越多（0.01→0.008→0.005→0.003）
static double MIN_DISTANCE = 4.0;    // 最小间距：越小点越密（8→5→4→3）
static int    BLOCK_SIZE = 3;      // Harris窗口：2或3更容易出更多点
static double HARRIS_K = 0.03;   // Harris k：越小越宽松（0.04→0.03→0.02）

// 可视化与显示
static int TARGET_H = 700;            // 两张图统一的显示高度
static bool SAVE_OUTPUT = true;       // 是否保存拼接图
/*———————————————————————————————————————*/

static Mat drawCorners_gFTT_Harris(const Mat& src) {
    Mat gray; cvtColor(src, gray, COLOR_BGR2GRAY);
    vector<Point2f> pts;

    // 用 goodFeaturesToTrack + Harris，既做阈值也做非极大值抑制
    goodFeaturesToTrack(
        gray, pts,
        MAX_CORNERS,      // 上限
        QUALITY_LEVEL,    // 越小→越多
        MIN_DISTANCE,     // 越小→越密
        Mat(),            // mask
        BLOCK_SIZE,       // 2 或 3 更敏感
        true,             // 使用 Harris 响应
        HARRIS_K          // Harris k
    );

    Mat vis = src.clone();
    for (const auto& p : pts) circle(vis, p, 3, Scalar(0, 0, 255), 1, LINE_AA);
    return vis;
}

static Mat resizeToHeight(const Mat& img, int h) {
    double s = h / (double)img.rows;
    Mat r; resize(img, r, Size(), s, s, INTER_AREA);
    return r;
}

int main() {
    Mat img1 = imread(PATH1), img2 = imread(PATH2);
    if (img1.empty() || img2.empty()) {
        cerr << "❌ 无法读取图片，请检查 PATH1/2。" << endl;
        return -1;
    }

    // 左右各自做角点
    Mat res1 = drawCorners_gFTT_Harris(img1);
    Mat res2 = drawCorners_gFTT_Harris(img2);

    // 等高、等比缩放后拼接
    res1 = resizeToHeight(res1, TARGET_H);
    res2 = resizeToHeight(res2, TARGET_H);

    Mat combined; hconcat(res1, res2, combined);
    putText(combined, "Image 1", Point(30, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
    putText(combined, "Image 2", Point(res1.cols + 30, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

    // 若太宽，自动整体再缩小到屏幕友好宽度
    int maxW = 1400;
    if (combined.cols > maxW) {
        double s = maxW / (double)combined.cols;
        resize(combined, combined, Size(), s, s, INTER_AREA);
    }

    namedWindow("Harris Comparison", WINDOW_NORMAL);
    resizeWindow("Harris Comparison", combined.cols, combined.rows);
    imshow("Harris Comparison", combined);
    if (SAVE_OUTPUT) imwrite("harris_comparison.jpg", combined);
    waitKey(0);
    return 0;
}
