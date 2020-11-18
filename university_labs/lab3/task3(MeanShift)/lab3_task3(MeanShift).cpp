#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat cimg, result;
void ResultImage(int mini, int maxi, Vec3b color, int n);

int main()
{
    Mat input_img = imread("roi_robotov_1.jpg");
    if (input_img.empty()) // Проверка на ошибки (загружено изображение или нет)
    {
        cout << "Could not open or find the image" << endl;
        system("pause"); // Ожидание нажатие кнопки
        return -1;
    }

    // Сегментация MeanShift
    Mat image_MeanShift;
    int spatial_radius = 60;
    int color_radius = 50;
    int pyramid_levels = 1;

    TermCriteria term_criteria{};
    term_criteria.maxCount = 5;
    term_criteria.epsilon = 1;
    term_criteria.type = TermCriteria::COUNT + TermCriteria::EPS;

    pyrMeanShiftFiltering(input_img, image_MeanShift, spatial_radius, color_radius, pyramid_levels, term_criteria);
    imshow("MeanShift", image_MeanShift); //проверка

    cvtColor(image_MeanShift, cimg, COLOR_BGR2HSV); // Преобразование из BGR в HSV
    imshow("MeanShift_HSV", cimg); // проверка
    // делаем две копии
    // buff для всей магии
    // result для рисования на нём
    Mat buff;
    buff = cimg.clone();
    result = input_img.clone();
    //imshow("img", img); // проверка

    // Пороговая фильтрация. Выбор диапазона для поиска лампы
    // lamp
    inRange(cimg, Vec3b(0, 0, 150), Vec3b(17, 23, 255), cimg);
    //imshow("cimg", cimg); // проверка

    /// Получение маски ядра для морфологии
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    /// Морфология
    erode(cimg, cimg, kernel, Point(-1, -1), 1); // Эрозия для устранения шумов
    dilate(cimg, cimg, kernel, Point(-1, -1), 3); // Дилатация для увеличения размеров

    /// Нахождение и рисование контура лампы
    ResultImage(50, 110, Vec3b(0, 255, 255), 1);

    /// Пороговая фильтрация. Выбор двух диапазононов для вычисления всех роботов нужного цвета
    /// red
    Mat a, b;
    inRange(buff, Vec3b(0, 10, 0), Vec3b(10, 255, 255), a);
    inRange(buff, Vec3b(160, 10, 0), Vec3b(179, 255, 255), b);
    cimg = a + b; // Сложение двух бинаризованных изображений и получение одного со всеми роботами нужного цвета
    // imshow("binary red", cimg); // проверка

    /// Нахождение и рисование контуров
    ResultImage(65, 130, Vec3b(0, 0, 255), 0);
    //imshow("conturs_red", result); // проверка

    /// green
    cimg = buff.clone();
    inRange(cimg, Vec3b(69, 50, 140), Vec3b(80, 255, 255), cimg);
    //imshow("binary green", cimg); // проверка

    ResultImage(60, 150, Vec3b(0, 255, 0), 0);
    //imshow("conturs_green", result); // проверка

    /// blue
    cimg = buff.clone();
    inRange(cimg, Vec3b(92, 50, 128), Vec3b(102, 255, 255), cimg);
    //imshow("binary blue", cimg); // проверка

    ResultImage(0, 150, Vec3b(255, 0, 0), 0);

    imshow("result", result);
    waitKey(0); // Ожидание нажатия клавиши
    destroyAllWindows(); // Уничтожение всех окон
    return 0;
}

void ResultImage(int mini, int maxi, Vec3b color, int n)
{
    Moments mnts, imnts, lmnts;
    vector<vector<Point>> cnts;
    // Поиск контуров объектов
    findContours(cimg, cnts, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    double d = 0;
    // Максимальное расстояние от робота до лампы (чтоб рисовать линию)
    double minum = 1000;
    /// Проход по всем контурам
    for (int i = 0; i < cnts.size(); i++)
    {
        // Если размер контура  не меньше минимума и не превышает максимум, то рисуем его и находим его центр масс, для избежания детектирования лампы
        if (cnts[i].size() > mini&& cnts[i].size() < maxi)
        {
            // Находим момент контура
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