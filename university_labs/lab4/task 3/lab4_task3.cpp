#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat input_img = imread("Car numbers.jpg", IMREAD_GRAYSCALE);
	if (input_img.empty()) // Проверка на ошибки (загружено изображение или нет)
	{
		cout << "Could not open or find the image" << endl;
		system("pause"); //ожидание нажатие кнопки
		return -1;
	}
	imshow("Car numbers", input_img);

	Mat templ_img = imread("A_letter.jpg", IMREAD_GRAYSCALE);
	if (templ_img.empty()) // Проверка на ошибки (загружено изображение или нет)
	{
		cout << "Could not open or find the image" << endl;
		system("pause"); //ожидание нажатие кнопки
		return -1;
	}
	imshow("A_letter", templ_img);

	input_img.convertTo(input_img, CV_32FC1);
	templ_img.convertTo(templ_img, CV_32FC1);

	/// Нахождение оптимальных значений для ускорения работы ДПФ
	Size dftSize;
	dftSize.width = getOptimalDFTSize(input_img.cols + templ_img.cols - 1);
	dftSize.height = getOptimalDFTSize(input_img.rows + templ_img.rows - 1);

	/// Создание и расширение холста (для получения одинакового числа гармоник)
	Mat dft_optimal(dftSize.height, dftSize.width, CV_32FC1, Scalar::all(0));
	Mat tempROI(dft_optimal, Rect(0, 0, input_img.cols, input_img.rows));
	input_img.copyTo(tempROI); // Копируем исходное изображение в ОИ от холста

	Mat templ_size_optimal(dftSize.height, dftSize.width, CV_32FC1, Scalar::all(0));
	Mat tempROI2(templ_size_optimal, Rect(0, 0, templ_img.cols, templ_img.rows));
	templ_img.copyTo(tempROI2); // Копируем исходное изображение в ОИ от холста

	Mat dft_img; // Образ Фурье изображения
	Mat dft_templ_img; // Образ шаблона Фурье
	dft(dft_optimal, dft_img, DFT_COMPLEX_OUTPUT);
	dft(templ_size_optimal, dft_templ_img, DFT_COMPLEX_OUTPUT);

	Mat dft_korr;// Корреляция изображения и шаблона
	mulSpectrums(dft_img, dft_templ_img, dft_korr, 0, 1);

	/// Корреляция изображения и шаблона до обрезки изображения
	Mat optimal_size_korr;
	optimal_size_korr.convertTo(optimal_size_korr, CV_32FC1);
	dft(dft_korr, optimal_size_korr, DFT_INVERSE | DFT_REAL_OUTPUT);

	/// Корреляция изображения и шаблона после обрезки изображения
	Mat korr(optimal_size_korr, Rect(0, 0, input_img.cols, input_img.rows));
	normalize(korr, korr, 0, 1, NORM_MINMAX);

	Point maxLoc; // Позиция наибольшего элемента
	minMaxLoc(korr, NULL, NULL, NULL, &maxLoc);
	cout << " maxLoc = " << maxLoc;

	Mat korr_thresh;
	threshold(korr, korr_thresh, 0.99, 1, THRESH_BINARY);// После нормализации наибольшее значение 1
	
	namedWindow("Tepml_thresh", WINDOW_AUTOSIZE);
	imshow("Tepml_thresh", korr_thresh);
	waitKey(0); // Ожидание нажатия клавиши
	destroyAllWindows(); // Уничтожение всех окон
	return 0;
}