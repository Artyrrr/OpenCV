#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>

using namespace cv;
using namespace std;


Mat KrasivSpektr(Mat magn)
{
	magn = magn(Rect(0, 0, magn.cols & -2, magn.rows & -2));
	/// Переставление квадрантов изображения Фурье
	int cx = magn.cols / 2;
	int cy = magn.rows / 2;

	Mat q0(magn, Rect(0, 0, cx, cy)); // верхний левый - разбиваем на квадранты
	Mat q1(magn, Rect(cx, 0, cx, cy)); // верхний правый
	Mat q2(magn, Rect(0, cy, cx, cy)); // нижний левый
	Mat q3(magn, Rect(cx, cy, cx, cy)); // нижний правый

	Mat tmp; // верхний левый меняем с нижним правым
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp); // верхний правый меняем с нижним левым
	q2.copyTo(q1);
	tmp.copyTo(q2);
	return magn;
}

int main() 
{
	Mat input_img = imread("Lenna.jpg", IMREAD_GRAYSCALE);
	if (input_img.empty()) // Проверка на ошибки (загружено изображение или нет)
	{
		cout << "Could not open or find the imagne" << endl;
		system("pause"); //ожидание нажатие кнопки
		return -1;
	}

	input_img.convertTo(input_img, CV_32FC1);
	
	/// Нахождение оптимальных значений для ускорения работы ДПФ
	Size dftSize;
	dftSize.width = getOptimalDFTSize(input_img.cols - 1);
	dftSize.height = getOptimalDFTSize(input_img.rows - 1); 

	/// Создание и расширение холста (для получения одинакового числа гармоник)
	Mat dft_optimal(dftSize.height, dftSize.width, CV_32FC1, Scalar::all(0));
	Mat tempROI(dft_optimal, Rect(0, 0, input_img.cols, input_img.rows));
	input_img.copyTo(tempROI); // Копируем исходное изображение в ОИ от холста

	Mat dft_img; // Образ Фурье 
	dft(dft_optimal, dft_img, DFT_COMPLEX_OUTPUT);

	Mat magn = KrasivSpektr(dft_img);

	Mat canvas(magn.rows, magn.cols, CV_8U, Scalar::all(0));
	circle(canvas, Point(canvas.cols / 2, canvas.rows / 2), 50, 255, -1); 
	Mat cutted_high;
	magn.copyTo(cutted_high, canvas);
	circle(magn, Point(magn.cols / 2, magn.rows / 2), 50, 0, -1); // Обрезка низких частот посредством круга

	namedWindow("Canvas", WINDOW_NORMAL);
	imshow("Canvas", canvas);

	Mat dft_cutted_high = KrasivSpektr(cutted_high);
	Mat dft_cutted_low = KrasivSpektr(magn);

	Mat img_cutted_high;
	Mat img_cutted_low;
	img_cutted_high.convertTo(img_cutted_high, CV_32FC1);
	img_cutted_low.convertTo(img_cutted_low, CV_32FC1);
	dft(dft_cutted_high, img_cutted_high, DFT_INVERSE | DFT_REAL_OUTPUT);
	dft(dft_cutted_low, img_cutted_low, DFT_INVERSE | DFT_REAL_OUTPUT);

	/// Обрезка изображения до исходного размера
	Mat img_high(img_cutted_high, Rect(0, 0, input_img.cols, input_img.rows));
	Mat img_low(img_cutted_low, Rect(0, 0, input_img.cols, input_img.rows));

	/// Для вывода imshow
	normalize(img_high, img_high, 0, 1, NORM_MINMAX);
	normalize(img_low, img_low, 0, 1, NORM_MINMAX);

	namedWindow("Lenna_high", WINDOW_NORMAL);
	imshow("Lenna_high", img_high);
	namedWindow("Lenna_low", WINDOW_NORMAL);
	imshow("Lenna_low", img_low);
	waitKey(0); // Ожидание нажатия клавиши
	destroyAllWindows(); // Уничтожение всех окон
	return 0;
}