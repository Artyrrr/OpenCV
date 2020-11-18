#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	Moments mnts;
	Mat imgBGR, imgHSV, cimg, kernel;

	string im = "./teplovizor/1.jpg";
	imgBGR = imread(im);
	if (imgBGR.empty()) // Проверка на ошибки (загружено изображение или нет)
	{
		cout << "Could not open or find the image" << endl;
		system("pause"); //ожидание нажатие кнопки
		return -1;
	}
	/// Показать изображение
	imshow("Input Image", imgBGR);

	cvtColor(imgBGR, imgHSV, COLOR_BGR2HSV); // Преобразование из BGR в HSV
	/// Пороговая фильтрация. Выбор диапазона для нахождения двигателя
	inRange(imgHSV, Vec3b(0, 50, 50), Vec3b(17, 255, 255), cimg);

	/// Получение маски ядра для морфологии
	kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	/// Сделать объект больше (иначе область определения будет не цельной)
	dilate(cimg, cimg, kernel, Point(-1, -1), 3);
	//	imshow("cimg", cimg); //проверка

	/// Поиск контура объекта
	vector<vector<Point>> cnts;
	findContours(cimg, cnts, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	//	imshow("contours", cimg); //проверка

	/// Поиск центра масс контура
	for (int i = 0; i < cnts.size(); i++)
	{
		if (cnts[i].size() > 30)
		{
			mnts = moments(cnts[i]);
			polylines(imgBGR, cnts[i], true, Vec3b(255, 255, 255), 2, 8);
			circle(imgBGR, Point(mnts.m10 / mnts.m00, mnts.m01 / mnts.m00), 5, Vec3b(0, 0, 0), -1);
		}
	}
	imshow("result", imgBGR);
	waitKey(0); // Ожидание нажатия клавиши
	destroyAllWindows(); // Уничтожение всех окон
	return 0;
}