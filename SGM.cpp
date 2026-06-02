#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // ---------------------------------------------------------
    // 1. 读取图片 (请确保你的电脑里有这两张图，且路径正确！)
    // 为了防止路径错误，建议新手先写绝对路径，例如 "D:\\images\\left.png"
    // 注意：Windows路径要用双斜杠 \\
    // ---------------------------------------------------------
    std::string pathL = "left.jpg"; // 如果图片在工程目录下
    std::string pathR = "right.jpg";

    // 强制以灰度模式读取 (解决 depth() != CV_8U 的问题)
    cv::Mat img_left = cv::imread(pathL, cv::IMREAD_GRAYSCALE);
    cv::Mat img_right = cv::imread(pathR, cv::IMREAD_GRAYSCALE);

    // ---------------------------------------------------------
    // 2. 详细的错误检查 (帮你定位问题)
    // ---------------------------------------------------------
    if (img_left.empty()) {
        std::cout << "【错误】无法读取左图！请检查路径: " << pathL << std::endl;
        // Visual Studio 常见坑：图片要放在 .vcxproj 文件旁边，或者用绝对路径
        return -1;
    }
    if (img_right.empty()) {
        std::cout << "【错误】无法读取右图！请检查路径: " << pathR << std::endl;
        return -1;
    }

    // 打印图片信息，让你看清楚
    std::cout << "左图尺寸: " << img_left.cols << "x" << img_left.rows << ", 通道: " << img_left.channels() << std::endl;
    std::cout << "右图尺寸: " << img_right.cols << "x" << img_right.rows << ", 通道: " << img_right.channels() << std::endl;

    // ---------------------------------------------------------
    // 3. 自动修复尺寸不一致 (解决 left.size() != right.size() 的问题)
    // ---------------------------------------------------------
    if (img_left.size() != img_right.size()) {
        std::cout << "【警告】两张图片尺寸不一致，正在强制缩放右图以匹配左图..." << std::endl;
        cv::resize(img_right, img_right, img_left.size());
    }

    // ---------------------------------------------------------
    // 4. SGBM 算法配置
    // ---------------------------------------------------------
    int minDisparity = 0;
    int numDisparities = 16 * 5; // 必须是16的倍数
    int blockSize = 3; // 必须是奇数

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        minDisparity,
        numDisparities,
        blockSize,
        8 * 1 * blockSize * blockSize,   // P1 (注意这里乘的是通道数1)
        32 * 1 * blockSize * blockSize,  // P2
        1, 63, 10, 100, 1
    );

    //// -------------------------未加入ROI--------------------------------
    //// 5. 计算视差 (已修复：加入计时功能)
    //// ---------------------------------------------------------
    //cv::Mat disp;
    //std::cout << "正在计算视差..." << std::endl;

    //// --- ⏱️ 计时开始 ---
    //int64 start = cv::getTickCount();

    //try {
    //    sgbm->compute(img_left, img_right, disp);
    //}
    //catch (cv::Exception& e) {
    //    std::cout << "【OpenCV 内部错误】: " << e.what() << std::endl;
    //    return -1;
    //}

    //// --- ⏱️ 计时结束 ---
    //int64 end = cv::getTickCount();

    //// 计算时间差 (秒) -> 转换为毫秒 (ms)
    //double time_sec = (end - start) / cv::getTickFrequency();
    //std::cout << ">>>>> SGM 耗时: " << time_sec * 1000 << " ms <<<<<" << std::endl;

    //std::cout << "计算完成！" << std::endl;
// ---------------------------------------------------------
    // 5. 计算视差 (修改版：加入 ROI 优化)
    // ---------------------------------------------------------
    cv::Mat disp;

    // --- 定义 ROI (只取图像中间 50% 的高度) ---
    // 模拟无人机避障场景：不需要看天，也不需要看脚下死角
    int y_start = img_left.rows / 4;      // 从 1/4 高度开始
    int height = img_left.rows / 2;       // 取 1/2 高度
    cv::Rect roi(0, y_start, img_left.cols, height);

    // 裁剪图片 (这是“零拷贝”操作，非常快)
    cv::Mat img_left_roi = img_left(roi);
    cv::Mat img_right_roi = img_right(roi);

    std::cout << "正在计算 ROI 视差..." << std::endl;

    // --- ⏱️ 计时开始 ---
    int64 start = cv::getTickCount();

    try {
        // 只计算 ROI 区域
        sgbm->compute(img_left_roi, img_right_roi, disp);
    }
    catch (cv::Exception& e) {
        std::cout << "【OpenCV 内部错误】: " << e.what() << std::endl;
        return -1;
    }

    // --- ⏱️ 计时结束 ---
    int64 end = cv::getTickCount();

    double time_sec = (end - start) / cv::getTickFrequency();
    std::cout << ">>>>> SGM + ROI 耗时: " << time_sec * 1000 << " ms <<<<<" << std::endl;

    // ---------------------------------------------------------
    // 6. 显示结果
    // ---------------------------------------------------------
    cv::Mat disp_vis;
    // 归一化以便显示 (SGM结果是16位，不处理看起来是全黑的)
    disp.convertTo(disp_vis, CV_8U, 255.0 / (numDisparities * 16.0));

    cv::imshow("Left Image", img_left);
    cv::imshow("Disparity Result", disp_vis);
    cv::waitKey(0);

    return 0;
}