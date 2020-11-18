#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int x = 0, y = 0, k = 0, p = 0;
double d, minum;

Moments mnts, imnts, lmnts;
Mat img, input_img, cimg, buff, kernel, result, lamp, red, green, blue, a, b;
vector<vector<Point>> cnts;

void result_img(int mini, int maxi, Vec3b color, int n);
void together();

int main()
{
	string im = "./roboti/roi_robotov.jpg";
	input_img = imread(im);
	if (input_img.empty()) // Проверка на ошибки (загружено изображение или нет)
	{
		cout << "Could not open or find the image" << endl;
		system("pause"); //ожидание нажатие кнопки
		return -1;
	}
	/// уменьшение окон
	resize(input_img, img, cv::Size(), 0.5, 0.5);
	resize(input_img, input_img, cv::Size(), 0.75, 0.75);

	cvtColor(input_img, cimg, COLOR_BGR2HSV); // Преобразование из BGR в HSV
	// делаем две копии
	// buff для всей магии
	// result для рисования на нём
	buff = cimg.clone();
	result = input_img.clone();
	imshow("img", img);
	// Пороговая фильтрация. Выбор диапазона для поиска лампы
	inRange(cimg, Vec3b(0, 0, 150), Vec3b(35, 12, 255), cimg);

	/// Получение маски ядра для морфологии
	kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

	/// Морфология
	erode(cimg, cimg, kernel, Point(-1, -1), 1); // эрозия для устранения шумов
	dilate(cimg, cimg, kernel, Point(-1, -1), 3); // дилатация для увеличения размеров

	/// нахождение и рисование контура лампы
	result_img(65, 110, Vec3b(0, 255, 255), 1);

	/// red
	cimg = buff.clone();
	erode(cimg, cimg, kernel, Point(-1, -1), 5); // эрозия для сглаживания
	//imshow("cimg", cimg); // проверка

	/// Пороговая фильтрация. Выбор диапазона двух диапазононов для вычисления всех роботов нужного цвета
	inRange(buff, Vec3b(0, 10, 0), Vec3b(10, 255, 255), a);
	inRange(buff, Vec3b(160, 10, 0), Vec3b(179, 255, 255), b);
	cimg = a + b; // Сложение двух бинаризованных изображений и получение одного со всеми роботами нужного цвета

	// imshow("binary red", cimg); // проверка

	/// Соединение роботов, которых перекрыла палка


	/// Нахождение и рисование контуров
	result_img(40, 120, Vec3b(0, 0, 255), 0);
	//imshow("conturs", result); // проверка

	/// green
	cimg = buff.clone();
	inRange(cimg, Vec3b(65, 50, 140), Vec3b(80, 255, 255), cimg);
	// imshow("binary green", cimg); // проверка

	result_img(17, 120, Vec3b(0, 255, 0), 0);

	/// blue
	cimg = buff.clone();
	inRange(cimg, Vec3b(92, 50, 128), Vec3b(102, 255, 255), cimg);
	// imshow("binary blue", cimg); // проверка


	result_img(0, 100, Vec3b(255, 0, 0), 0);

	imshow("result", result);
	waitKey(0); // Ожидание нажатия клавиши
	destroyAllWindows(); // Уничтожение всех окон
	return 0;
}

void result_img(int mini, int maxi, Vec3b color, int n)
{
	// Поиск контуров объектов
	findContours(cimg, cnts, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	d = 0;
	/// максимальное расстояние от робота до лампы (чтоб рисовать линию)
	minum = 1000;
	/// Проход по всем контурам
	for (int i = 0; i < cnts.size(); i++)
	{
		// если размер контура  не меньше минимума и не превышает максимум, то рисуем его и находим его центр масс, для избежания детектирования лампы
		if (cnts[i].size() > mini&& cnts[i].size() < maxi)
		{
			// находим момент контура
			mnts = moments(cnts[i]);
			// если 0, то ищутся роботы, иначе лампу
			if (n == 0)
			{
				// Нахождение расстояния от центра масс робота до центра масс лампы
				d = sqrt((lmnts.m10 / lmnts.m00 - mnts.m10 / mnts.m00) * (lmnts.m10 / lmnts.m00 - mnts.m10 / mnts.m00) +
					(lmnts.m01 / lmnts.m00 - mnts.m01 / mnts.m00) * (lmnts.m01 / lmnts.m00 - mnts.m01 / mnts.m00));
				if (minum > d)
				{
					// Обновление максимального расстояния
					minum = d;
					imnts = mnts;
				}
			}
			else if (n == 1)
			{
				/// Моменты лампы
				lmnts = mnts; // для запоминания
				imnts = mnts; // для рисования
			}

			polylines(result, cnts[i], true, color, 2, 8); // Рисование контура
		}
	}
	circle(result, Point(imnts.m10 / imnts.m00, imnts.m01 / imnts.m00), 5, Vec3b(0, 0, 0), -1); // Рисование центра масс
}

