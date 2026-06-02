#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
	cv::Mat img = cv::imread("img1.jpg"); //Mat是Matrix矩阵的缩写，读取文件不要加空格
	int rows = img.rows;
	int cols = img.cols;

	for (int i = 0; i < rows; i++)
	{
		uchar* data = img.ptr<uchar>(i);//.ptr是Pointer指针的缩写
		for (int j = 0; j < cols; j++)
		{
			int b = data[3 * j];
			int g = data[3 * j + 1];
			int r = data[3 * j + 2];

			int gray = 0.11 * b + 0.59 * g + 0.30 * r;
			data[3 * j] = gray;
			data[3 * j + 1] = gray;
			data[3 * j + 2] = gray;
		}
	}

	cv::imshow("My Gray Image", img);
	cv::waitKey(0);
}