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

void CamCalibProc(Mat& camera_matrix, Mat& distortio);
bool SaveCameraCalibration(string calib_param, Mat camera_matrix, Mat distortio);
void CameraCalibration(vector<Mat> saved_images, Size board_size, Mat& camera_matrix, Mat& distortio);
void GetChessboardCorners(vector<Mat> saved_images, vector<vector<Point2f>>& corners, bool show_results);
inline void CreateKnownBoardPosition(Size boardSize, vector<Point3f>& corners);
bool LoadCameraCalibration(string calib_param, Mat& camera_matrix, Mat& distortio);

static const float square_calib_length = 0.02905f; // Размер квадрата калибровочной доски в метрах
static const Size chessboard_size = Size(6, 9); // Размер калибровочной доски

int main()
{
    Mat camera_matrix = Mat::eye(3, 3, CV_64F);
    Mat distortio_coef; // Коэффициенты дисторсии

    CamCalibProc(camera_matrix, distortio_coef); // Запуск калибровки камеры
    LoadCameraCalibration("ILoveCameraCalibration", camera_matrix, distortio_coef); // Загрузка старых или новых настроек

    return 0;
}


/// Заполнение массива точек углов калибровочной доски, углы создаются в системе координат калибровочной доски
inline void CreateKnownBoardPosition(Size boardSize, vector<Point3f>& corners)
{
    for (int i = 0; i < boardSize.height; i++)
        for (int j = 0; j < boardSize.width; j++)
            corners.push_back(Point3f(j * square_calib_length, i * square_calib_length, 0));
}

/// Нахождение углов квадратов калибровочной доски для каждого калибровочного изображения
void GetChessboardCorners(vector<Mat> saved_images, vector<vector<Point2f>>& corners, bool show_results = false)
{
    for (vector <Mat>::iterator iter = saved_images.begin(); iter != saved_images.end(); iter++)
    {
        vector < Point2f> corners_temp;
        bool found = findChessboardCorners(*iter, Size(chessboard_size.height, chessboard_size.width), corners_temp, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        if (found) corners.push_back(corners_temp);
        if (show_results)
        {
            drawChessboardCorners(*iter, Size(chessboard_size.height, chessboard_size.width), corners_temp, found);
            imshow("Chess_corners", *iter);
            waitKey(0);
        }
    }
}


/// Производит калибровку камеры после нажатия клавиши enter по набору картинок полученных в ходе калибровки
void CameraCalibration(vector<Mat> saved_images, Size board_size, Mat& camera_matrix, Mat& distortio)
{
    vector<vector<Point2f>> chessboard_corners; // Точки углов калибровочной доски для каждого изображения
    GetChessboardCorners(saved_images, chessboard_corners, false);

    vector<vector<Point3f>> world_space_corner_points(1); // Массив из 1 массива точек углов калибровочной доски в своей СК
    CreateKnownBoardPosition(board_size, world_space_corner_points[0]); // Заполнение массива
    /// Заполнение массива для каждого изображения одинаковыми значениями
    world_space_corner_points.resize(chessboard_corners.size(), world_space_corner_points[0]);

    vector<Mat> rvec, tvec; // Массив векторов вращения и переноса изображения для каждого изображения
    distortio = Mat::zeros(8, 1, CV_64F); // Заготовка для коэффициентов дисторсии
}

bool SaveCameraCalibration(string calib_param, Mat camera_matrix, Mat distortio)
{
    ofstream to_file(calib_param); // Связывание потока вывода с файлом
    if (to_file) {
        to_file << "camera_matrix.rows = " << endl;
        to_file << camera_matrix.rows << endl;
        to_file << "camera_matrix.cols = " << endl;
        to_file << camera_matrix.cols << endl;

        for (int r = 0; r < camera_matrix.rows; r++)
            for (int c = 0; c < camera_matrix.cols; c++)
                to_file << " camera_matrix_val(" << r << "," << c << ") = " << endl << camera_matrix.at<double>(r, c) << endl;

        to_file << "distor_matrix.rows = " << endl;
        to_file << distortio.rows << endl;
        to_file << "distor_matrix.cols = " << endl;
        to_file << distortio.cols << endl;

        for (int r = 0; r < distortio.rows; r++)
            for (int c = 0; c < distortio.cols; c++)
                to_file << "distor_matrix_val(" << r << "," << c << ") = " << endl << distortio.at<double>(r, c) << endl;

        to_file.close(); // Закрытие файла
        return true;
    }
    return false;
}

bool LoadCameraCalibration(string calib_param, Mat& camera_matrix, Mat& distortio)
{
    ifstream from_file(calib_param);
    if (from_file) {
        unsigned int rows;
        unsigned int columns;
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

        // Коэффициенты дисторсии
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

void CamCalibProc(Mat& camera_matrix, Mat& distortio)
{
    const unsigned int frames_per_second = 20;
    Mat frame; // Захваченный кадр
    Mat frame_with_chess_corners; // Кадр с обозначенными углами калибровочной доски
    vector<Mat> saved_images;
    VideoCapture vid(0);
    if (vid.isOpened() == false) {
        std::cout << "error: Webcam does not connect\n";
        return;
    }

    namedWindow("Webcam", WINDOW_NORMAL);
    while (true)
    {
        if (!vid.read(frame))
            break;
        vector<Vec2f> found_pts;
        bool found = false;

        found = findChessboardCorners(frame, chessboard_size, found_pts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        frame.copyTo(frame_with_chess_corners);
        drawChessboardCorners(frame_with_chess_corners, chessboard_size, found_pts, found);
        if (found)        
            imshow("Webcam", frame_with_chess_corners);
        else                         
            imshow("Webcam", frame);
        char character = waitKey(1000 / frames_per_second);

        switch (character)
        {
        case ' ':  // Сохранение фотографии кнопкой пробел
            if (found)
            {
                Mat temp;

                frame.copyTo(temp);
                saved_images.push_back(temp);
            }
            break;
        case 13: // Запуск алгоритм вычисления параметров камеры и сохранить результаты при кол-ве изображений более 15
            if (saved_images.size() > 15) {
                CameraCalibration(saved_images, chessboard_size, camera_matrix, distortio);
                SaveCameraCalibration("ILoveCameraCalibration", camera_matrix, distortio);
            }
            break;
        case 27:
            return;
            break;
        }
    }
}
