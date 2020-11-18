#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void DrawMap();
void Rangefinder(const Mat frame, Mat map);

int main() 
{
	DrawMap();
	Mat clear_map = imread("./example/map.jpg");
	Mat frame;
	VideoCapture video("./example/calib_1.avi");
	while (true) 
	{
		video.read(frame);
		Mat clean = clear_map.clone();
		Rangefinder(frame, clean);
		namedWindow("Frame", WINDOW_NORMAL);
		imshow("Frame", frame);
		imshow("Map", clean);
		if (waitKey(30) >= 0)
		{
			break;
		}
	}
	return 0;
}

/// Функция создания карты
void DrawMap()
{
	Mat map = Mat(640, 480, CV_8UC3);
	for (int i = 0; i <= map.rows; i += 40)
	{
		line(map, Point(0, i), Point(map.cols, i), Scalar(0), 1, LINE_AA);
	}
	for (int i = 0; i <= map.cols; i += 40)
	{
		line(map, Point(i, 0), Point(i, map.rows), Scalar(0), 1, LINE_AA);
	}
	imwrite("./example/map.jpg", map);
	imshow("DrawMap", map); // проверка
}

/// Функция рисования меток на карте
void Rangefinder(const Mat frame, Mat map)
{
	namedWindow("Control", WINDOW_NORMAL); // Создание окна с ползунками

	int iLowH = 60;
	int iHighH = 90;
	int iLowS = 50;
	int iHighS = 200;
	int iLowV = 90;
	int iHighV = 200;

	/// Создание трекбара
	createTrackbar("LowH", "Control", &iLowH, 255); // Hue (0 - 255)
	createTrackbar("HighH", "Control", &iHighH, 179);

	createTrackbar("LowS", "Control", &iLowS, 255); // Saturation (0 - 255)
	createTrackbar("HighS", "Control", &iHighS, 255);

	createTrackbar("LowV", "Control", &iLowV, 255); // Value (0 - 255)
	createTrackbar("HighV", "Control", &iHighV, 255);

	Mat temp(frame.size(), CV_8U);
	Mat temp2(frame.size(), CV_8U);
	cvtColor(frame, temp, COLOR_BGR2HSV);
	inRange(temp, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), temp2);
	imshow("Control", temp2);
	for (int i = 0; i < temp2.rows; i++)
	{
		for (int j = 0; j < temp2.cols; j++)
		{
			if (temp2.at<uchar>(i, j) != 0)
			{
				double z_ = 19.24; // Расстояние от матрицы камеры до фокуса
				double y = 14.57;  // Расстояние от луча лазера до центра матрицы камеры по вертикали
				double k = 0.045; // Коэффициент подобия
				double x_ = (j - temp2.cols / 2) * k; // Расстояние от центра матрицы до точки матрицы (в которой луч света от точки пространства пересёк матрицу) по горизонтали 
				double y_ = (i - temp2.rows / 2) * k; // Расстояние от центра матрицы до точки матрицы (в которой луч света от точки пространства пересёк матрицу) по вертикали
				/// Применение подобия треугольников
				int z = static_cast<int>(z_ * y / y_); 
				int x = static_cast<int>(x_ * z / z_);
				drawMarker(map, Point(map.cols / 2 - 4 * x, 4 * z), Scalar(0, 255, 0), 2, 4);
			}
		}
	}
}
