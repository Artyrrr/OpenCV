#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void ClassifyCoins(Mat money, Mat copper, Mat nickel, Mat& result);

int main()
{
	Mat nickel = imread("Ni.jpg");
	if (((nickel.empty())))
	{
		cout << "Could not open or find some images for nikel" << endl;
		system("pause");
		return -1;
	}

	Mat copper = imread("Cu.jpg");
	if ((copper.empty()))
	{
	cout << "Could not open or find some images for copper" << endl;
	system("pause");
	return -1;
	}


	Mat money = imread("Money.png");
	if ((money.empty()))
	{
		cout << "Could not open or find some images for money" << endl;
		system("pause");
		return -1;
	}

	Mat result;

	ClassifyCoins(money, copper, nickel, result);
	imshow("Classify Coins", result);
	waitKey();
	destroyAllWindows();
}

void ClassifyCoins(Mat money, Mat copper, Mat nickel, Mat& result)
{
	/// Создание маски для меди
	Mat copper_mask;
	int thresh_h_copper = 8;
	int thresh_s_copper = 14;
	int thresh_v_copper = 40;
	cvtColor(copper, copper, COLOR_BGR2HSV);
	inRange(copper, Scalar(thresh_h_copper, thresh_s_copper, thresh_v_copper), Scalar(179, 255, 255), copper_mask);

	/// Создание маски для никеля
	Mat nickel_mask;
	int thresh_h_nickel = 16;
	int thresh_s_nickel = 4;
	int thresh_v_nickel = 18;
	cvtColor(nickel, nickel, COLOR_BGR2HSV);
	inRange(nickel, Scalar(thresh_h_nickel, thresh_s_nickel, thresh_v_nickel), Scalar(179, 255, 255), nickel_mask);

	Mat gray_money;
	result = money.clone();
	cvtColor(result, gray_money, COLOR_BGR2GRAY);
	vector<Vec3f> circles;
	HoughCircles(gray_money, circles, HOUGH_GRADIENT, 1, 20, 300, 54); // Функция поиска кругов Хафа

	vector<Mat> masks;
	for (vector<Vec3f>::iterator i = circles.begin(); i != circles.end(); i++)
	{
		Mat mask = Mat::zeros(money.size(), CV_8U);
		Vec3f circle_param = *i;
		circle(mask, Point(circle_param[0], circle_param[1]), circle_param[2], Scalar(255), FILLED);
		masks.push_back(mask);
	}

	/// Задание параметров для гистограммы
	int hue = 30; 
	int saturation = 35;
	int size_histogram[] = {hue, saturation};
	float h_ranges[] = {0, 180};
	float s_ranges[] = {0, 256};
	const float* ranges[] = {h_ranges, s_ranges};

	/// Вычисление гистограммы
	Mat copper_hist;
	Mat nickel_hist;
	int channels[] = {0, 1};
	calcHist(&copper, 1, channels, copper_mask, copper_hist, 2, size_histogram, ranges, true, false);
	calcHist(&nickel, 1, channels, nickel_mask, nickel_hist, 2, size_histogram, ranges, true, false);

	/// Перевод в модель HSV
	Mat hsv;
	cvtColor(result, hsv, COLOR_BGR2HSV);
	vector<Mat> hists;
	for (int i = 0; i < circles.size(); i++)
	{
		Mat histogram;
		calcHist(&hsv, 1, channels, masks[i], histogram, 2, size_histogram, ranges, true, false);
		hists.push_back(histogram);
	}

	vector<double> copper_compare;
	vector<double> nickel_compare;

	for (vector<Mat>::iterator i = hists.begin(); i != hists.end(); i++)
	{
		double corr = compareHist(*i, copper_hist, HISTCMP_BHATTACHARYYA);
		copper_compare.push_back(corr);
	}

	for (vector<Mat>::iterator i = hists.begin(); i != hists.end(); i++)
	{
		double corr = compareHist(*i, nickel_hist, HISTCMP_BHATTACHARYYA);
		nickel_compare.push_back(corr);
	}

	vector<double> compliance;
	for (int i = 0; i < circles.size(); i++)
		compliance.push_back(copper_compare[i] / nickel_compare[i]);
	/// Сравнение корреляции гистограммы монеты с гистограммой изображения какой-то из монет
	for (int i = 0; i < circles.size(); i++)
	{
		if (compliance[i] > 0.8)
		{
			circle(result, Point(circles[i][0], circles[i][1]), circles[i][2], Scalar(255, 0, 0), 2);
		}
		else
		{
			circle(result, Point(circles[i][0], circles[i][1]), circles[i][2], Scalar(0, 255, 0), 2);
		}
	}
}





