#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

int main() {
    // 1. ��ȡͼƬ (��ʽ���� cv::)
    cv::Mat img = cv::imread("test.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) return -1;

    // ������С
    cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2));

    // ==========================================
    // ���� 1: ģ�⡰�߶ȿռ䡱 (��˹ģ��)
    // ==========================================
    cv::Mat g1, g2, g3;
    // ģ���һ�������
    cv::GaussianBlur(img, g1, cv::Size(0, 0), 1.6);
    cv::GaussianBlur(img, g2, cv::Size(0, 0), 2.0);
    cv::GaussianBlur(img, g3, cv::Size(0, 0), 2.5);

    cv::imwrite("step1_gaussian_1.jpg", g1);
    cv::imwrite("step1_gaussian_2.jpg", g2);
    cv::imwrite("step1_gaussian_3.jpg", g3);

    // ==========================================
    // ���� 2: ģ�⡰��˹��� (DoG)��
    // ==========================================
    cv::Mat dog1, dog2;
    cv::subtract(g2, g1, dog1);
    cv::subtract(g3, g2, dog2);

    // ��һ���Ա���ӻ�
    cv::Mat dog1_vis, dog2_vis;
    cv::normalize(dog1, dog1_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(dog2, dog2_vis, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::imwrite("step2_dog_1.jpg", dog1_vis);
    cv::imwrite("step2_dog_2.jpg", dog2_vis);

    // ==========================================
    // ���� 3: ���չؼ��� (���� OpenCV SIFT)
    // ==========================================
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    sift->detect(img, keypoints);

    cv::Mat img_keypoints;
    cv::drawKeypoints(img, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imwrite("step3_final_result.jpg", img_keypoints);

    return 0;
}