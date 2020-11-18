#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat MyFourier(const Mat img) 
{
	double Pi = 3.141;
	/// Матрицы мнимых и вещественных коэффициентов для Фурье преобразования
	Mat real_koef(img.cols, img.cols, CV_32FC1);
	Mat imagninary_koef(img.cols, img.cols, CV_32FC1);
	for (int i = 0; i <= img.cols - 1; i++) 
	{
		for (int j = 0; j <= img.cols - 1; j++) 
		{
			real_koef.at<float>(i, j) = cos(-2 * Pi * i * j / img.cols);
			imagninary_koef.at<float>(i, j) = sin(-2 * Pi * i * j / img.cols);
		}
	}

	/// Матрицы мнимых и вещественных частей образа Фурье
	Mat dft_real(img.size(), CV_32FC1);
	Mat dft_imagninary(img.size(), CV_32FC1);
	dft_real = img * real_koef;
	dft_imagninary = -1 * img * imagninary_koef;

	/// Всё это в одно двухканальное изображение
	Mat dft_img(img.size(), CV_32FC2);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) 
		{
			dft_img.at<Vec2f>(i, j)[0] = dft_real.at<float>(i, j);
			dft_img.at<Vec2f>(i, j)[1] = dft_imagninary.at<float>(i, j);
		}


	Mat my_dft(img.size(), CV_32FC2, Scalar::all(0));
	for (int u = 0; u < img.rows; u++) 
	{
		for (int w = 0; w < img.cols; w++) 
		{
			float real = 0, imagninary = 0;
			for (int n = 0; n < img.rows; n++) 
			{
				complex<float> value(dft_img.at<Vec2f>(n, w)[0], dft_img.at<Vec2f>(n, w)[1]);
				float phase = -2 * Pi * u * n / img.rows - arg(value);
				float magnn = abs(value);
				real += magnn * cos(phase);
				imagninary += magnn * sin(phase);
			}
			my_dft.at<Vec2f>(u, w) = Vec2f(real, imagninary);
		}
	}
	return my_dft;
}

/// Функция рисования образа Фурье и перемещение квадрантов (1\4 изображения) так, чтобы низкие частоты оказались в центре
Mat KrasivSpektr(const Mat dft_img) 
{
	/// Деление многоканального изображение на 2 одноканальных
	Mat mvbegin[2];
	split(dft_img, mvbegin);

	/// Нахождение магнитуды для каждого элемента образа Фурье
	Mat magn(dft_img.rows, dft_img.cols, CV_32FC1);
	magnitude(mvbegin[0], mvbegin[1], magn); // Определение магнитуды

	/// Переход к логарифмическому масштабу
	magn += Scalar::all(1);
	log(magn, magn);// Натуральный логарифм для каждого элемента
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
	normalize(magn, magn, 0, 1, NORM_MINMAX); // Для вывода imshow
	return magn;
}

Mat Laplas(void) // Фильтр Лапласа
{
	Mat kernel(3, 3, CV_32FC1);
	kernel.at<float>(0, 0) = 0.0f;
	kernel.at<float>(0, 1) = 1.0f;
	kernel.at<float>(0, 2) = 0.0f;
	kernel.at<float>(1, 0) = 1.0f;
	kernel.at<float>(1, 1) = -4.0f;
	kernel.at<float>(1, 2) = 1.0f;
	kernel.at<float>(2, 0) = 0.0f;
	kernel.at<float>(2, 1) = 1.0f;
	kernel.at<float>(2, 2) = 0.0f;
	return kernel;
}

Mat BoxFilter(void) // Box фильтр
{
	Mat kernel(3, 3, CV_32FC1);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			kernel.at<float>(i, j) = 1.0f / 9.0f;
	return kernel;
}

Mat SobelHorizon(void) // Фильтр Собеля по горизонтали
{
	Mat kernel(3, 3, CV_32FC1);
	kernel.at<float>(0, 0) = -1.0f;
	kernel.at<float>(0, 1) = 0.0f;
	kernel.at<float>(0, 2) = 1.0f;
	kernel.at<float>(1, 0) = -2.0f;
	kernel.at<float>(1, 1) = 0.0f;
	kernel.at<float>(1, 2) = 2.0f;
	kernel.at<float>(2, 0) = -1.0f;
	kernel.at<float>(2, 1) = 0.0f;
	kernel.at<float>(2, 2) = 1.0f;
	return kernel;
}

Mat SobelVerical(void) // Фильтр Собеля по вертикали
{
	Mat kernel(3, 3, CV_32FC1);
	kernel.at<float>(0, 0) = -1.0f;
	kernel.at<float>(0, 1) = -2.0f;
	kernel.at<float>(0, 2) = -1.0f;
	kernel.at<float>(1, 0) = 0.0f;
	kernel.at<float>(1, 1) = 0.0f;
	kernel.at<float>(1, 2) = 0.0f;
	kernel.at<float>(2, 0) = 1.0f;
	kernel.at<float>(2, 1) = 2.0f;
	kernel.at<float>(2, 2) = 1.0f;
	return kernel;
}

int main() 
{
	Mat input_img = imread("fourier.png", IMREAD_GRAYSCALE);
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
	
	/// Вызов фильтров
	Mat laplas = Laplas();
	Mat box = BoxFilter();
	Mat sobel_horizon = SobelHorizon();
	Mat sobel_vertical = SobelVerical();

	/// Создание и расширение холста (для получения одинакового числа гармоник)
	Mat dft_optimal(dftSize.height, dftSize.width, CV_32FC1, Scalar::all(0));
	Mat tempROI(dft_optimal, Rect(0, 0, input_img.cols, input_img.rows));
	input_img.copyTo(tempROI); // Копируем исходное изображение в ОИ от холста

	Mat laplas_optimal(dftSize.height, dftSize.width, CV_32FC1, Scalar::all(0));
	Mat tempROI2(laplas_optimal, Rect(0, 0, 3, 3));
	laplas.copyTo(tempROI2); // Копируем исходное изображение в ОИ от холста


	Mat box_optimal(dftSize.height, dftSize.width, CV_32FC1, Scalar::all(0));
	Mat tempROI3(box_optimal, Rect(0, 0, 3, 3));
	box.copyTo(tempROI3); // Копируем исходное изображение в ОИ от холста

	Mat sobel_horizon_optimal(dftSize.height, dftSize.width, CV_32FC1, Scalar::all(0));
	Mat tempROI4(sobel_horizon_optimal, Rect(0, 0, 3, 3));
	sobel_horizon.copyTo(tempROI4); // Копируем исходное изображение в ОИ от холста

	Mat sobel_vertical_optimal(dftSize.height, dftSize.width, CV_32FC1, Scalar::all(0));
	Mat tempROI5(sobel_vertical_optimal, Rect(0, 0, 3, 3));
	sobel_vertical.copyTo(tempROI5); // Копируем исходное изображение в ОИ от холста

	/// Прямое дискретное преобразование Фурье
	Mat dft_img(input_img.rows, input_img.rows, CV_32FC2);
	dft(dft_optimal, dft_img, DFT_COMPLEX_OUTPUT);
	Mat dft_laplas(3, 3, CV_32FC2);
	dft(laplas_optimal, dft_laplas, DFT_COMPLEX_OUTPUT);
	Mat dft_box(3, 3, CV_32FC2);
	dft(box_optimal, dft_box, DFT_COMPLEX_OUTPUT);
	Mat dft_sobel_horizon(3, 3, CV_32FC2);
	dft(sobel_horizon_optimal, dft_sobel_horizon, DFT_COMPLEX_OUTPUT);
	Mat dft_sobel_vertical(3, 3, CV_32FC2);
	dft(sobel_vertical_optimal, dft_sobel_vertical, DFT_COMPLEX_OUTPUT);

	//Mat my_fourier = MyFourier(input_img); // Для запуска моей функции ДПФ

	Mat magn1 = KrasivSpektr(dft_img);
	Mat magn2 = KrasivSpektr(dft_laplas);
	Mat magn3 = KrasivSpektr(dft_box);
	Mat magn4 = KrasivSpektr(dft_sobel_horizon);
	Mat magn5 = KrasivSpektr(dft_sobel_vertical);
	//Mat magn6 = KrasivSpektr(my_fourier); // Для запуска моей функции ДПФ

	/// Результат
	namedWindow("Fourier magn", WINDOW_NORMAL);
	imshow("Fourier magn", magn1);

	namedWindow("Laplas magn", WINDOW_NORMAL);
	imshow("Laplas magn", magn2);

	namedWindow("Box magn", WINDOW_NORMAL);
	imshow("Box magn", magn3);

	namedWindow("Sobel Vertical magn", WINDOW_NORMAL);
	imshow("Sobel Vertical magn", magn4);

	namedWindow("Sobel Horizon magn", WINDOW_NORMAL);
	imshow("Sobel Horizon magn", magn5);

	//namedWindow("MyFourier magn", WINDOW_NORMAL); 
	//imshow("MyFourier magn", magn6); // Для запуска моей функции ДПФ

	/// Возвращение в наше измерение. Обратное преобразование Фурье

	Mat img_laplas_dft(laplas_optimal.rows, laplas_optimal.cols, CV_32FC2);
	Mat img_box_dft(box_optimal.rows, box_optimal.cols, CV_32FC2);
	Mat img_sobel_horizon_dft(sobel_horizon_optimal.rows, sobel_horizon_optimal.cols, CV_32FC2);
	Mat img_sobel_vertical_dft(sobel_vertical_optimal.rows, sobel_vertical_optimal.cols, CV_32FC2);

	/// Перемножение спектров
	mulSpectrums(dft_img, dft_laplas, img_laplas_dft, 0, 0);
	mulSpectrums(dft_img, dft_box, img_box_dft, 0, 0);
	mulSpectrums(dft_img, dft_sobel_horizon, img_sobel_horizon_dft, 0, 0);
	mulSpectrums(dft_img, dft_sobel_vertical, img_sobel_vertical_dft, 0, 0);

	/// Обратное дискретное преобразование Фурье
	Mat img_laplas_temp;
	img_laplas_temp.convertTo(img_laplas_temp, CV_32FC1);
	dft(img_laplas_dft, img_laplas_temp, DFT_INVERSE | DFT_REAL_OUTPUT);
	Mat img_laplas(img_laplas_temp, Rect(0, 0, input_img.cols, input_img.rows));

	Mat img_box_temp;
	img_box_temp.convertTo(img_box_temp, CV_32FC1);
	dft(img_box_dft, img_box_temp, DFT_INVERSE | DFT_REAL_OUTPUT);
	Mat img_box(img_box_temp, Rect(0, 0, input_img.cols, input_img.rows));

	Mat img_sobel_h_temp;
	img_sobel_h_temp.convertTo(img_sobel_h_temp, CV_32FC1);
	dft(img_sobel_horizon_dft, img_sobel_h_temp, DFT_INVERSE | DFT_REAL_OUTPUT);
	Mat img_sobel_horizon(img_sobel_h_temp, Rect(0, 0, input_img.cols, input_img.rows));

	Mat img_sobel_v_temp;
	img_sobel_v_temp.convertTo(img_sobel_v_temp, CV_32FC1);
	dft(img_sobel_vertical_dft, img_sobel_v_temp, DFT_INVERSE | DFT_REAL_OUTPUT);
	Mat img_sobel_vertical(img_sobel_v_temp, Rect(0, 0, input_img.cols, input_img.rows));

	/// Для вывода imshow
	normalize(input_img, input_img, 0, 1, NORM_MINMAX);
	normalize(img_laplas, img_laplas, 0, 1, NORM_MINMAX);
	normalize(img_box, img_box, 0, 1, NORM_MINMAX);
	normalize(img_sobel_vertical, img_sobel_vertical, 0, 1, NORM_MINMAX);
	normalize(img_sobel_horizon, img_sobel_horizon, 0, 1, NORM_MINMAX);

	namedWindow("Reverse Sobel Horizon", WINDOW_NORMAL);
	imshow("Reverse Sobel Horizon", img_sobel_horizon); // Результат обратного преобразования
	waitKey(0); // Ожидание нажатия клавиши
	destroyAllWindows(); // Уничтожение всех окон
	return 0;
}