#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

void box_filter(const Mat& image, Mat& img_box, cv::Size ksize);
int compare(const Mat& image1, const Mat& image2);

void box_filter(const Mat& image, Mat& img_box, cv::Size ksize)
{
	image.copyTo(img_box);
	int shiftX = ksize.height / 2;
	int shiftY = ksize.width / 2;

	for (int i = shiftX; i < image.rows - shiftX; i++)
	{
		for (int j = shiftY; j < image.cols - shiftY; j++)
		{
			Rect rp(j - shiftY, i - shiftX, 3, 3);
			Mat roi = image(rp);
			unsigned int sum = 0;
			for (int i = 0; i < rp.height; i++)
			{
				for (int j = 0; j < rp.width; j++)
				{
					sum += (unsigned int)(roi.at<uchar>(i, j));
				}
			}
			img_box.at<uchar>(i, j) = uchar(round(sum / (double(ksize.width) * double(ksize.height))));
		}
	}
}

int compare(const Mat& image1, const Mat& image2)
{
	int temp = 0;
	for (int i = 0; i < image1.rows; i++)
	{
		for (int j = 0; j < image1.cols; j++)
		{
			if (image1.at<uchar>(i, j) == image2.at<uchar>(i, j))
				temp++;
		}
	}
	return (int)(temp * 100 / (image1.cols * image1.rows));
}

int main()
{
	TickMeter time_blur;
	TickMeter time_box;
	time_blur.reset();
	time_box.reset();

	Mat img_blur;
	Mat img_box;

	Mat img = imread("Milky_Way.jpg", 0);
	imshow("Image", img);

	time_blur.start();
	blur(img, img_blur, Size(3, 3));
	time_blur.stop();

	imshow("Standart blur", img_blur);
	cout << "TimeSec of blur filter: " << time_blur.getTimeSec() << endl;

	time_box.start();
	box_filter(img, img_box, Size(3, 3));
	time_box.stop();
	
	imshow("box_filter", img_box);
	cout << "TimeSec of box filter: " << time_box.getTimeSec() << endl;

	cout << "Difference in TimeSec: " << time_box.getTimeSec() - time_blur.getTimeSec() << endl; //разница по времени выполнения функций

	cout << "Similarity percentage:" << compare(img_blur, img_box) << endl; //процент схожести изображений
	waitKey(0);
	destroyAllWindows();
	return 0;
}