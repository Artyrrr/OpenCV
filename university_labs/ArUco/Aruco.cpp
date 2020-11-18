#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace aruco;

Mat DrawCube(Mat frame, const vector<Point2f> corners, const Vec3d rvec, const Vec3d tvec, const Mat camera_matrix, const Mat distortio, const float sq_side);
void InitPeaks(Mat& peaks, const double square_side);
bool LoadCameraCalibration(string calib_param, Mat& camera_matrix, Mat& distortio );
int StartWebcamMonitoring(const Mat& camera_matrix, const Mat& distortio , const float square_marker_lengths);

static const float square_marker_length = 0.132f; // Размер стороны квадрата маркера в метрах
static const Ptr<Dictionary> markerDictionary = getPredefinedDictionary(PREDEFINED_DICTIONARY_NAME::DICT_4X4_100);

int main()
{
    Mat camera_matrix = Mat::eye(3, 3, CV_64F); // Матрица с единицами по диагонали и остальными нулями, одноканальная, 64f - формат данных, содержит коэффициенты камеры
    Mat distortio_coef; // Коэффициенты дисторсии
    LoadCameraCalibration("Camera_сalibration", camera_matrix, distortio_coef); // Загрузить старые настройки или только что перезаписанные
    StartWebcamMonitoring(camera_matrix, distortio_coef, square_marker_length); // Начать обнаружение маркеров
    return 0;
}

/// Рисует куб на найденном маркере
Mat DrawCube(Mat frame, const vector<Point2f> corners, const Vec3d rvec, const Vec3d tvec, const Mat camera_matrix, const Mat distortio , const float sq_side)
{
    Mat peaks = Mat::zeros(8, 4, CV_64F);
    InitPeaks(peaks, sq_side);
    Mat rotation_matrix = Mat::zeros(3, 3, CV_64F);
    Rodrigues(rvec, rotation_matrix); // Преобразование rvec в матрицу поворота камеры относительно маркера
    Mat trans_matrix = Mat::zeros(4, 4, CV_64F);

    /// Матрица перехода
    for (uint i = 0; i < rotation_matrix.rows; i++)
        for (uint j = 0; j < rotation_matrix.cols; j++)
            trans_matrix.at<double>(i, j) = rotation_matrix.at<double>(i, j);

    /// Вектор переноса
    trans_matrix.at<double>(0, 3) = tvec[0];
    trans_matrix.at<double>(1, 3) = tvec[1];
    trans_matrix.at<double>(2, 3) = tvec[2];
    trans_matrix.at<double>(3, 3) = 1;

    Mat cube_trans = Mat::zeros(8, 4, CV_64F);
    Mat coordinate_cube = Mat::zeros(8, 4, CV_64F);

    cube_trans = peaks * trans_matrix.t(); // Получение матрицы перехода для вершин куба относительно собственной СК куба

    coordinate_cube = Mat::zeros(cube_trans.rows, 2, CV_32S); // Преобразование в координаты изображения u, v
    for (uint i = 0; i < cube_trans.rows; i++) {
        double x_ = (cube_trans.at<double>(i, 0) / cube_trans.at<double>(i, 2));
        double y_ = (cube_trans.at<double>(i, 1) / cube_trans.at<double>(i, 2));
        coordinate_cube.at<int>(i, 0) = fabs(x_ * camera_matrix.at<double>(0, 0) - camera_matrix.at<double>(0, 2));
        coordinate_cube.at<int>(i, 1) = fabs(y_ * camera_matrix.at<double>(1, 1) - camera_matrix.at<double>(1, 2));
    }

    /// Перенос вершин куба на вершины маркера, при этом нижние углы совпадают с углами маркера, а верхние определяются матрицей u,v
    for (uint i = 0; i < 4; i++)
    {
        int temp_x = coordinate_cube.at<int>(i, 0);
        int temp_y = coordinate_cube.at<int>(i, 1);
        int err_x = corners[i].x - temp_x;
        int err_y = corners[i].y - temp_y;
        coordinate_cube.at<int>(i, 0) += err_x;
        coordinate_cube.at<int>(i, 1) += err_y;
        coordinate_cube.at<int>(i + 4, 0) += err_x;
        coordinate_cube.at<int>(i + 4, 1) += err_y;
    }

    /// Отрисовка вертикальных линий
    for (uint i = 0; i < 4; i++)
        line(frame, Point(coordinate_cube.at<int>(i, 0), coordinate_cube.at<int>(i, 1)), Point(coordinate_cube.at<int>(i + 4, 0), coordinate_cube.at<int>(i + 4, 1)), Scalar(255 / (3 * i + 1), 255 / (5 * i + 1), 255 / (2 * i + 1)), 4, 5);

    /// Отрисовка горизонтальных линий
    for (uint i = 0; i < coordinate_cube.rows - 1; i++)
    {
        if (i == 3)
        {
            line(frame, Point(coordinate_cube.at<int>(3, 0), coordinate_cube.at<int>(3, 1)), Point(coordinate_cube.at<int>(0, 0), coordinate_cube.at<int>(0, 1)), Scalar(255, 0, 255), 4, 5);
            line(frame, Point(coordinate_cube.at<int>(7, 0), coordinate_cube.at<int>(7, 1)), Point(coordinate_cube.at<int>(4, 0), coordinate_cube.at<int>(4, 1)), Scalar(255, 255, 0), 4, 5);
            continue;
        }
        line(frame, Point(coordinate_cube.at<int>(i, 0), coordinate_cube.at<int>(i, 1)),
            Point(coordinate_cube.at<int>(i + 1, 0), coordinate_cube.at<int>(i + 1, 1)), Scalar(255, 255 / (7 * i + 1), 0), 4, 5);
    }
    return frame; // Возвращение изображения с нарисованным кубом
}

/// Создание матрицы вершин куба по x, y, z
void InitPeaks(Mat& peaks, const double square_side)
{
    peaks.at<double>(0, 0) = 1;
    peaks.at<double>(2, 1) = 1;
    peaks.at<double>(2, 3) = 1;
    peaks.at<double>(3, 0) = 1;
    peaks.at<double>(3, 1) = 1;
    peaks.at<double>(4, 0) = 1;
    peaks.at<double>(4, 2) = 1;
    peaks.at<double>(5, 2) = 1;
    peaks.at<double>(6, 1) = 1;
    peaks.at<double>(6, 2) = 1;
    for (int i = 0; i < 4; i++)
        peaks.at<double>(7, i) = 1;

    peaks *= square_side;
    for (int i = 0; i < peaks.rows; i++)
        peaks.at<double>(i, 3) = 1;
}

bool LoadCameraCalibration(string calib_param, Mat& camera_matrix, Mat& distortio)
{
    ifstream from_file(calib_param);
    if (from_file)
    {
        uint16_t rows;
        uint16_t columns;
        char buff[50]; // Буфер промежуточного хранения считываемого из файла текста
        from_file.getline(buff, 50);
        from_file >> rows;
        from_file.getline(buff, 50);
        from_file.getline(buff, 50);
        from_file >> columns;
        from_file.getline(buff, 50);
        from_file.getline(buff, 50);

        camera_matrix = Mat(Size(columns, rows), CV_64F);

        for (int r = 0; r < rows; r++)
            for (int c = 0; c < columns; c++)
            {
                double read = 0.0f;
                from_file >> read;
                camera_matrix.at<double>(r, c) = read;
                cout << camera_matrix.at<double>(r, c) << "\n";
                from_file.getline(buff, 50);
                from_file.getline(buff, 50);
            }

        /// Коэффициенты дисторсии
        from_file >> rows;
        from_file.getline(buff, 50);
        from_file.getline(buff, 50);
        from_file >> columns;

        distortio = Mat::zeros(rows, columns, CV_64F);

        for (int r = 0; r < rows; r++)
            for (int c = 0; c < columns; c++)
            {
                from_file.getline(buff, 50);
                from_file.getline(buff, 50);
                double read = 0.0f;
                from_file >> read;
                distortio.at<double>(r, c) = read;
                cout << distortio.at<double>(r, c) << "\n";
            }
        from_file.close();
        return true;
    }
    return false;
}

int StartWebcamMonitoring(const Mat& camera_matrix, const Mat& distortio , const float square_marker_lengths)
{
    Mat frame;
    vector<int> marker_Ids;
    vector<vector<Point2f>> markerCorners, rejectedCandidates;
    Ptr<DetectorParameters> parameters = DetectorParameters::create();
    vector<Vec3d> rvec, tvec;

    VideoCapture vid(0, CAP_DSHOW);
    if (vid.isOpened() == false)
    {
        cout << "error: Webcam doesn't connect\n";
        return(0);
    }

    namedWindow("Webcam", WINDOW_NORMAL);

    while (true)
    {
        if (!vid.read(frame))
            break;

        /// Выдаёт засечённые на поданном изображении маркеры по используемому словарю, на выход положение углов каждого маркера, id каждого,параметры поиска, отклонённые кандидаты
        detectMarkers(frame, markerDictionary, markerCorners, marker_Ids, parameters, rejectedCandidates);

        /// Определение положение осей в виде матриц поворота и сдвига маркеров относительно начала координат
        estimatePoseSingleMarkers(markerCorners, square_marker_length, camera_matrix, distortio , rvec, tvec);

        /// Выводит оси на изображение
        for (int i = 0; i < marker_Ids.size(); i++)
        {
            drawAxis(frame, camera_matrix, distortio , rvec[i], tvec[i], 0.07f); // Последний параметр - длина осей в метрах
            frame = DrawCube(frame, markerCorners[i], rvec[i], tvec[i], camera_matrix, distortio, 0.03f);
        }
        imshow("Webcam", frame);
        if (waitKey(30) >= 0) break; // Прерывание по нажатию клавишы
    }
    return 1;
}