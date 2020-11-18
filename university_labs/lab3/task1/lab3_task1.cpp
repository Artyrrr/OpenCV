#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat img, thimg, cimg;

int main()
{
	Moments mnts, imnts, lmnts;
	int color = 0;

	Mat img = imread("./allababah/ig_0.jpg", color);
	if (img.empty()) // Проверка на ошибки (загружено изображение или нет)
	{
		cout << "Could not open or find the image" << endl;
		system("pause"); // Ожидание нажатие кнопки
		return -1;
	}
	/// Показать изображение
	imshow("Input Image", img);
	/// Установить двоичный режим
	threshold(img, thimg, 230, 255, THRESH_BINARY); // Простая двоичная пороговая фильтрация
	cvtColor(img, img, COLOR_GRAY2BGR);

	/// Получение маски ядра для морфологии
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

	//	erode(thimg, cimg, kernel, Point(-1, -1), 2);
	//	imshow("erode", cimg);

	/// Сделать объект больше (иначе область определения будет не цельной)
	dilate(thimg, cimg, kernel, Point(-1, -1), 4);
	imshow("dilate", cimg);

	/// Поиск и рисование контура объекта
	vector<vector<Point>> cnts;
	findContours(cimg, cnts, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	imshow("contours", cimg);

	polylines(img, cnts, true, Vec3b(0, 255, 0), 2, 8); // Рисуем контур
	imshow("polyline", img);
	/// Поиск центра масс контура
	for (int i = 0; i < cnts.size(); i++)
	{
		if (cnts[i].size() > 10)
		{
			mnts = moments(cnts[i]);
			circle(img, Point(mnts.m10 / mnts.m00, mnts.m01 / mnts.m00), 5, Vec3b(0, 0, 255), -1); 
		}
	}
	imshow("result", img);
	waitKey(0); // Ожидание нажатия клавиши
	destroyAllWindows(); // Уничтожение всех окон
	return 0;
}