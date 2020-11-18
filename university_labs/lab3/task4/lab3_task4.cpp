#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;


int main()
{
	Mat  timg, cimg;
	int color = 0;

	Mat input_img = imread("./gk/gk.jpg", color);
	Mat img = imread("./gk/gk_tmplt.jpg", color);
	if (img.empty()) // Проверка на ошибки (загружено изображение или нет)
	{
		cout << "Could not open or find the image" << endl;
		system("pause"); //ожидание нажатие кнопки
		return -1;
	}
	resize(input_img, input_img, cv::Size(), 0.75, 0.75); // Уменьшение окна
	/// Установить двоичный режим
	threshold(input_img, cimg, 240, 255, THRESH_BINARY_INV); // Обратная двоичная пороговая фильтрация

	/// Поиск и рисование контура объектов
	vector<vector<Point>> cnts, tnts;
	findContours(cimg, cnts, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	cvtColor(input_img, input_img, COLOR_GRAY2BGR);
	threshold(img, timg, 240, 255, THRESH_BINARY);
	cvtColor(img, img, COLOR_GRAY2BGR);

	/// Поиск и рисование контура маски
	findContours(timg, tnts, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	polylines(img, tnts[0], true, Vec3b(0, 255, 0), 2, 8);

	/// Сравнение маски с каждым объектом и выделение разными цветами в зависимости от похожести
	for (int i = 0; i < cnts.size(); i++)
	{
		float diff = matchShapes(cnts[i], tnts[0], CONTOURS_MATCH_I2, 0);
		if (abs(diff) < 0.5)
		{
			polylines(input_img, cnts[i], true, Vec3b(0, 255, 0), 2, 8);
		}
		else
		{
			polylines(input_img, cnts[i], true, Vec3b(0, 0, 255), 2, 8);
		}
	}
	imshow("input_img", input_img);
	imshow("template", img);
	waitKey(0); // Ожидание нажатия клавиши
	destroyAllWindows(); // Уничтожение всех окон
	return 0;
}
