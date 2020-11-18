#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat Skelet(Mat frame_orig);
void ZhangSuenOne(Mat& frame, int& deleted_pix_per_step);
void ZhangSuenTwo(Mat& frame, int& deleted_pix_per_step);
void LinesTogether(Mat& frame, Mat& img);
void DeleteLines(Mat& frame, vector<Vec4i>& lines_P, vector<double>& k, vector<double>& b);
void Association(Mat& frame, Mat& img, vector<Vec4i>& lines_P, Mat& next_frame, vector<double>& k, vector<double>& b);

int main() 
{
    Mat frame;
    VideoCapture vid("./example/1.avi");
    while (true) 
    {
        vid.read(frame);
        Mat lines = Skelet(frame);
        LinesTogether(lines, frame);
        namedWindow("Track", WINDOW_AUTOSIZE);
        imshow("Track", frame);
        if (waitKey(30) >= 0)
            break; 
    }
}

void ZhangSuenOne(Mat& frame, int& deleted_pix_per_step)
{
    Mat delete_pixel = Mat::zeros(frame.rows, frame.cols, CV_8U);
    for (int i = 1; i < frame.rows - 1; i++) 
    {
        uchar* top_row = frame.ptr<uchar>(i - 1);
        uchar* current_row = frame.ptr<uchar>(i);
        uchar* bottom_row = frame.ptr<uchar>(i + 1);
        uchar* pixel_for_delete = delete_pixel.ptr<uchar>(i);
        for (int j = 1; j < frame.cols - 1; j++) 
        {
            uint current = j;
            uint right = j + 1;
            uint left = j - 1;
            uchar P_1_1 = top_row[current];
            uchar P_1_2 = current_row[right];
            uchar P_1_3 = bottom_row[current];
            uchar P_2_1 = bottom_row[current];
            uchar P_2_2 = current_row[left];
            uchar P_2_3 = current_row[right];
            if (frame.at<uchar>(i, j) == 255) 
            {
                if ((P_1_1 == 0) || (P_1_2 == 0) || (P_1_3 == 0)) 
                {
                    if ((P_2_1 == 0) || (P_2_2 == 0) || (P_2_3 == 0)) 
                    {
                        int transition = 0;
                        int white_pix_num = -1;
                        if ((top_row[current] == 0) && (top_row[right] == 255))
                            transition++;
                        if ((top_row[right] == 0) && (current_row[right] == 255))
                            transition++;
                        if ((current_row[right] == 0) && (bottom_row[right] == 255))
                            transition++;
                        if ((bottom_row[right] == 0) && (bottom_row[current] == 255))
                            transition++;
                        if ((bottom_row[current] == 0) && (bottom_row[left] == 255))
                            transition++;
                        if ((bottom_row[left] == 0) && (current_row[left] == 255))
                            transition++;
                        if ((current_row[left] == 0) && (top_row[left] == 255))
                            transition++;
                        if ((top_row[left] == 0) && (top_row[current] == 255))
                            transition++;

                        for (int k = i - 1; k <= i + 1; k++) 
                        {
                            uchar* ptr = frame.ptr<uchar>(k);
                            for (int m = j - 1; m <= j + 1; m++)
                                if (ptr[m] == 255)
                                    white_pix_num++;
                        }

                        if ((transition == 1) && ((white_pix_num >= 2) && (white_pix_num <= 6)))
                        {
                            pixel_for_delete[j] = 255;
                            deleted_pix_per_step++;
                        }

                    }
                }
            }
        }
    }

    for (int i = 0; i < delete_pixel.rows; i++)
    {
        uchar* cur_pix_ptr = frame.ptr<uchar>(i);
        for (int j = 0; j < delete_pixel.cols; j++)
            if (delete_pixel.at<uchar>(i, j) == 255)
                cur_pix_ptr[j] = 0;
    }
}

void ZhangSuenTwo(Mat& frame, int& deleted_pix_per_step) 
{
    Mat delete_pixel = Mat::zeros(frame.rows, frame.cols, CV_8U);
    for (int i = 1; i < frame.rows - 1; i++)
    {
        uchar* top_row = frame.ptr<uchar>(i - 1);
        uchar* current_row = frame.ptr<uchar>(i);
        uchar* bottom_row = frame.ptr<uchar>(i + 1);
        uchar* pixel_for_delete = delete_pixel.ptr<uchar>(i);
        for (int j = 1; j < frame.cols - 1; j++)
        {
            uint right = j + 1;
            uint current = j;
            uint left = j - 1;
            uchar P_1_1 = top_row[current];
            uchar P_1_2 = current_row[right];
            uchar P_1_3 = current_row[left]; 
            uchar P_2_1 = bottom_row[current];
            uchar P_2_2 = current_row[left];
            uchar P_2_3 = top_row[current];
            if (frame.at<uchar>(i, j) == 255) 
            {
                if ((P_1_1 == 0) || (P_1_2 == 0) || (P_1_3 == 0))
                {
                    if ((P_2_1 == 0) || (P_2_2 == 0) || (P_2_3 == 0))
                    {
                        int transition = 0;
                        int white_pix_num = -1;
                        if ((top_row[current] == 0) && (top_row[right] == 255))
                            transition++;
                        if ((top_row[right] == 0) && (current_row[right] == 255))
                            transition++;
                        if ((current_row[right] == 0) && (bottom_row[right] == 255))
                            transition++;
                        if ((bottom_row[right] == 0) && (bottom_row[current] == 255))
                            transition++;
                        if ((bottom_row[current] == 0) && (bottom_row[left] == 255))
                            transition++;
                        if ((bottom_row[left] == 0) && (current_row[left] == 255))
                            transition++;
                        if ((current_row[left] == 0) && (top_row[left] == 255))
                            transition++;
                        if ((top_row[left] == 0) && (top_row[current] == 255))
                            transition++;

                        for (int k = i - 1; k <= i + 1; k++) 
                        {
                            uchar* ptr = frame.ptr<uchar>(k);
                            for (int m = j - 1; m <= j + 1; m++)
                                if (ptr[m] == 255)
                                    white_pix_num++;
                        }

                        if ((transition == 1) && ((white_pix_num >= 2) && (white_pix_num <= 6)))
                        {
                            pixel_for_delete[j] = 255;
                            deleted_pix_per_step++;
                        }

                    }
                }
            }
        }
    }

    for (int i = 0; i < delete_pixel.rows; i++)
    {
        uchar* cur_pix_ptr = frame.ptr<uchar>(i);
        for (int j = 0; j < delete_pixel.cols; j++)
            if (delete_pixel.at<uchar>(i, j) == 255)
                cur_pix_ptr[j] = 0;
    }
}

Mat Skelet(Mat frame_orig) 
{
        Mat frame;
        Mat frame_hsv;
        cvtColor(frame_orig, frame_hsv, COLOR_BGR2HSV); // Для получения бинарной маски
        inRange(frame_hsv, Scalar(0, 0, 100), Scalar(255, 135, 255), frame);
        while (1)
        {
            int deleted_pix_per_step = 0;
            ZhangSuenOne(frame, deleted_pix_per_step);
            ZhangSuenTwo(frame, deleted_pix_per_step);
            if (deleted_pix_per_step == 0)
                return frame;
        }
}

void DeleteLines(Mat& frame, vector<Vec4i>& lines_P, vector<double>& k, vector<double>& b) 
{
    for (size_t i = 0; i < lines_P.size(); i++) 
    {
        Vec4i l = lines_P[i];
        /// Нахождение коэффициентов для линейной функции
        k[i] = static_cast<double>(l[1] - l[3]) / static_cast<double>(l[0] - l[2]); 
        b[i] = l[1] - k[i] * l[0];
    }


    for (size_t i = 0; i < lines_P.size(); i++)
        for (size_t j = 0; j < lines_P.size(); j++) 
        {
            if (i == j) continue;
            // Удаление прямых, которые которые близки по параметрам к k и b во избежании повтора
            if ((abs(k[i] - k[j]) < 0.15) && (abs(b[i] - b[j]) < 150)) 
            {
                lines_P.erase(lines_P.begin() + i);
                vector<Vec4i>(lines_P).swap(lines_P);
                k.erase(k.begin() + i);
                vector<double>(k).swap(k);
                b.erase(b.begin() + i);
                vector<double>(b).swap(b);
                if (i == lines_P.size()) break;
            }
        }

    for (size_t i = 0; i < lines_P.size(); i++) 
    {
        if ((lines_P[i][1] >= 478) && (lines_P[i][3] >= 478)) 
        {
            lines_P.erase(lines_P.begin() + i);
            vector<Vec4i>(lines_P).swap(lines_P);
            k.erase(k.begin() + i);
            vector<double>(k).swap(k);
            b.erase(b.begin() + i);
            vector<double>(b).swap(b);
            if (i == lines_P.size()) break;
        }
    }
}

void Association(Mat& frame, Mat& img, vector<Vec4i>& lines_P, Mat& next_frame, vector<double>& k, vector<double>& b)
{
    /// Объединение прямых в одну ломаную
    vector<Point> pairs;
    for (size_t i = 0; i < lines_P.size() - 1; i++) 
    {
        vector<double> distances;
        vector<int> edges;
        Vec4i current_line = lines_P[i];
        for (size_t j = 0; j < lines_P.size(); j++) 
        {
            if (j == i) continue;
            Vec4i other_line = lines_P[j];
            vector<double> dist(4);
            dist[0] = sqrt(pow((current_line[0] - other_line[0]), 2) + pow((current_line[1] - other_line[1]), 2));
            dist[1] = sqrt(pow((current_line[0] - other_line[2]), 2) + pow((current_line[1] - other_line[3]), 2));
            dist[2] = sqrt(pow((current_line[2] - other_line[0]), 2) + pow((current_line[3] - other_line[1]), 2));
            dist[3] = sqrt(pow((current_line[2] - other_line[2]), 2) + pow((current_line[3] - other_line[3]), 2));
            auto smallest = min_element(begin(dist), end(dist));
            int min_index = distance(begin(dist), smallest);
            distances.push_back(dist[min_index]);
            edges.push_back(min_index);
        }
        auto smallest_dist = min_element(begin(distances), end(distances));
        int min_index_dist = distance(begin(distances), smallest_dist);
        int next_line_index = -1;

        if (min_index_dist < i)
            next_line_index = min_index_dist;
        else
            next_line_index = min_index_dist + 1;

        /// Проверка: была ли такая пара прямых
        for (int s = 0; s < (pairs.size()); s++) 
        {
            int wrong_index = -1;
            for (int g = 0; g < pairs.size(); g++) 
            {
                if (((static_cast<int>(pairs[g].x) == i) && (static_cast<int>(pairs[g].y) == next_line_index)) || ((static_cast<int>(pairs[g].x) == next_line_index) && ((static_cast<int>(pairs[g].y) == i)))) 
                {
                    if (static_cast<int>(pairs[g].x) == i)
                        wrong_index = pairs[g].y;
                    if (static_cast<int>(pairs[g].y) == i)
                        wrong_index = pairs[g].x;
                }
            }
            /// Добавление максимального значения получившееся линии и нахождение индекса следующей линии для соединения
            if (wrong_index >= 0)
            {
                distances[wrong_index] = INT32_MAX;
                auto smallest_dist = min_element(begin(distances), end(distances));
                min_index_dist = distance(begin(distances), smallest_dist);
                int temp_index = min_index_dist;
                next_line_index = -1;
                min_index_dist < i ? next_line_index = min_index_dist : next_line_index = min_index_dist + 1;
            }
        }
        /// Поиск кратчайшего расстояния до каждой следующей прямой - наиболее близкой к текущей
        double x_common = (b[next_line_index] - b[i]) / (k[i] - k[next_line_index]);
        double y_common = k[i] * x_common + b[i];
        Point temp(i, next_line_index);
        pairs.push_back(temp);
        if (edges[min_index_dist] == 0) 
        {
            lines_P[i][0] = x_common;
            lines_P[next_line_index][0] = x_common;
            lines_P[i][1] = y_common;
            lines_P[next_line_index][1] = y_common;
        }
        if (edges[min_index_dist] == 1) 
        {
            lines_P[i][0] = x_common;
            lines_P[next_line_index][2] = x_common;
            lines_P[i][1] = y_common;
            lines_P[next_line_index][3] = y_common;
        }
        if (edges[min_index_dist] == 2) 
        {
            lines_P[i][2] = x_common;
            lines_P[next_line_index][0] = x_common;
            lines_P[i][3] = y_common;
            lines_P[next_line_index][1] = y_common;
        }
        if (edges[min_index_dist] == 3) 
        {
            lines_P[i][2] = x_common;
            lines_P[next_line_index][2] = x_common;
            lines_P[i][3] = y_common;
            lines_P[next_line_index][3] = y_common;
        }
    }

    for (size_t i = 0; i < lines_P.size(); i++)
    {
        Vec4i l = lines_P[i];
        line(next_frame, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 1, LINE_AA);
    }

    for (size_t i = 0; i < lines_P.size(); i++) 
    {
        Vec4i l = lines_P[i];
        line(img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 128, 0), 2, LINE_AA);
    }
}

void LinesTogether(Mat& frame, Mat& img) 
{
    Mat next_frame = Mat::zeros(frame.rows, frame.cols, CV_8U);
    vector<Vec4i> lines_P;
    HoughLinesP(frame, lines_P, 1, CV_PI / 360, 30, 45, 70); // Вероятностное преобразование Хафа
    vector<double> k(lines_P.size());
    vector<double> b(lines_P.size());
    vector<double> top_points_y;
    DeleteLines(frame, lines_P, k, b);
    Association(frame, img, lines_P, next_frame, k, b);
    namedWindow("Skelet", WINDOW_AUTOSIZE);
    imshow("Skelet", next_frame); // Проверка. Рисование следующего кадра
    waitKey(2000);
    destroyWindow("Skelet");
}