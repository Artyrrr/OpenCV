//с доп. заданием - прозрачный квадрат см. addWeighted(square, alpha, Milky_Way, beta, 0.0, square);
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	Mat Milky_Way = imread("Milky_Way.jpg"); // Считывание файла изображения
	if (Milky_Way.empty()) // Проверка на ошибки (загружено изображение или нет)
	{
		cout << "Could not open or find the image" << endl;
		system("pause"); //ожидание нажатие кнопки
		return -1;
	}

	double y[100];
	double angle[100];
	float x;
	for (x = 0; x < 100; x++)
	{
		y[(int)floor(x)] = 400 + 70 * sin(x / 4); // Положение синусоиды в окне по координате y
		angle[(int)floor(x)] = 35 * cos(x / 4); // Задание угла (поворот "роботелеги")
	}

	Mat img(500, 500, CV_8UC3);
	img.setTo(255);
	for (int x = 0; x < 99; x++)
	{
		line(Milky_Way, Point(Milky_Way.cols * x / 100, y[x]), Point(Milky_Way.cols * (x + 1) / 100, y[x + 1]), CV_RGB(0, 0, 255)); // Задание вида синусоиды
	}

	String windowName = "Окно"; //Название окна

	namedWindow(windowName); // Создание окна

	imshow(windowName, Milky_Way); // Показ картинки в созданном окне
	for (int x = 0; x < 99; x++)
	{
		double alpha = 0.5; double beta;
		Point2f pts[4];
		Mat square;
		Milky_Way.copyTo(square);
		RotatedRect parameters1 = RotatedRect(Point2f(square.cols * x / 100, y[x]), Size2f(50, 70), angle[x]); // Поворот прямоугольника
		parameters1.points(pts);
		Point pts1[4];
		for (int t = 0; t < 4; t++)
			pts1[t] = (pts[t]);
		fillConvexPoly(square, pts1, 4, CV_RGB(0, 0, 0), 4, 0);
		beta = (1.0 - alpha);
		addWeighted(square, alpha, Milky_Way, beta, 0.0, square);
		waitKey(100);
		imshow(windowName, square); // Показ изображения внутри созданного окна
	}

	waitKey(0); // Ожидание нажатия клавиши в окне

	destroyWindow(windowName); // Уничтожение всех окон

	return 0;
}