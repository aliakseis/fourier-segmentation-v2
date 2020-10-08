// fourier-segmentation-v2.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "tswdft2d.h"


#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <opencv2/photo.hpp>

#include <opencv2/plot.hpp>

#include <iostream>
#include <map>


//const int IMAGE_DIMENSION = 800;
const int IMAGE_DIMENSION = 512;

enum { WINDOW_DIMENSION = 32 };

const auto visualizationRows = IMAGE_DIMENSION - WINDOW_DIMENSION + 1;
const auto visualizationCols = IMAGE_DIMENSION - WINDOW_DIMENSION + 1;


void DemoShow(const cv::Mat& src, const char* caption)
{
    auto copy = src.clone();
    resize(copy, copy, cv::Size(256, 256));
    imshow(caption, copy);
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    using namespace cv;

    const auto& transformed = *static_cast<std::vector<std::complex<float>>*>(userdata);

    enum { HALF_SIZE = WINDOW_DIMENSION / 2 };

    if (event == cv::EVENT_MOUSEMOVE)
    {
        const auto xx = std::min(std::max(x - HALF_SIZE, 0), IMAGE_DIMENSION - WINDOW_DIMENSION);
        const auto yy = std::min(std::max(y - HALF_SIZE, 0), IMAGE_DIMENSION - WINDOW_DIMENSION);

        const auto sourceOffset = yy * visualizationCols + xx;


        std::map<float, float> ordered;

        for (int j = 1; j < WINDOW_DIMENSION/* * WINDOW_DIMENSION*/; ++j)
        {
            if (j / WINDOW_DIMENSION > WINDOW_DIMENSION / 2 || j % WINDOW_DIMENSION > WINDOW_DIMENSION / 2)
                continue;

            const auto amplitude = std::abs(transformed[sourceOffset * WINDOW_DIMENSION * WINDOW_DIMENSION + j]);
            const auto freq = hypot(j / WINDOW_DIMENSION, j % WINDOW_DIMENSION);
            if (freq > 2)
                ordered[freq] = std::max(ordered[freq], amplitude);
        }

        Mat data_x; //(1, 51, CV_64F);
        Mat data_y; //(1, 51, CV_64F);

        data_x.push_back(0.);
        data_y.push_back(0.);

        for (auto& v : ordered)
        {
            data_x.push_back(double(v.first));
            data_y.push_back(double(v.second));
        }

        //cv::normalize(data_y, data_y);


        Mat plot_result;

        Ptr<plot::Plot2d> plot = plot::Plot2d::create(data_x, data_y);
        //plot->render(plot_result);
        //imshow("The plot", plot_result);

        plot->setShowText(false);
        plot->setShowGrid(false);
        plot->setPlotBackgroundColor(Scalar(255, 200, 200));
        plot->setPlotLineColor(Scalar(255, 0, 0));
        plot->setPlotLineWidth(2);
        plot->setInvertOrientation(true);
        plot->render(plot_result);

        imshow("The plot", plot_result);

        cv::Mat magI(WINDOW_DIMENSION, WINDOW_DIMENSION, CV_32FC1);
        for (int j = 0; j < WINDOW_DIMENSION * WINDOW_DIMENSION; ++j)
        {
            const auto amplitude = std::abs(transformed[sourceOffset * WINDOW_DIMENSION * WINDOW_DIMENSION + j]);
            magI.at<float>(j / WINDOW_DIMENSION, j % WINDOW_DIMENSION) = amplitude;
        }

        magI += Scalar::all(1);                    // switch to logarithmic scale
        log(magI, magI);

        normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a

        auto magIcopy = magI.clone();
        cv::resize(magIcopy, magIcopy, cv::Size(512, 512));

        imshow("original spectrum magnitude", magIcopy);

        // rearrange the quadrants of Fourier image  so that the origin is at the image center
        int cx = magI.cols / 2;
        int cy = magI.rows / 2;
        Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
        Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
        Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
        Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

        //DemoShow(q0, "q0");
        //DemoShow(q1, "q1");
        //DemoShow(q2, "q2");
        //DemoShow(q3, "q3");

        Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
        q2.copyTo(q1);
        tmp.copyTo(q2);
                                                // viewable image form (float between values 0 and 1).
        //imshow("Input Image", I);    // Show the result

        cv::resize(magI, magI, cv::Size(512, 512));

        imshow("spectrum magnitude", magI);



        //cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;

        //cv::Rect roi(cv::Point(std::max(x - HALF_SIZE, 0), std::max(y - HALF_SIZE, 0)), 
        //    cv::Point(std::min(x + HALF_SIZE - 1, IMAGE_DIMENSION), std::min(y + HALF_SIZE - 1, IMAGE_DIMENSION)));


        //displayFourier((*m)(roi));

        //auto copy = (*m)(roi).clone();
        //auto line = copy.reshape(0, 1);
        //const auto mean = cv::mean(line);
        //line -= mean;
        //cv::normalize(line, line);

        //displayFourier(copy);
    }
}


int main(int argc, char *argv[])
{
    /*Read Image*/
    cv::Mat src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    cv::Mat img;
    src.convertTo(img, CV_32F);


    cv::resize(img, img, cv::Size(IMAGE_DIMENSION, IMAGE_DIMENSION), 0, 0, cv::INTER_LANCZOS4);

    const auto kernel_size = 3;
    cv::Mat dst;
    cv::GaussianBlur(img, dst, cv::Size(kernel_size, kernel_size), 0, 0, cv::BORDER_DEFAULT);
    //const auto filtered = dst.clone();

    dst += 1.;
    cv::log(dst, dst);

    cv::Mat stripeless;
    GaussianBlur(dst, stripeless, cv::Size(63, 1), 0, 0, cv::BORDER_DEFAULT);

    //cv::Mat funcFloat = (dst - stripeless + 8) * 16;
    cv::Mat funcFloat = dst - stripeless;
    normalize(funcFloat, funcFloat, 0, 255, cv::NORM_MINMAX);
    cv::Mat func;
    funcFloat.convertTo(func, CV_8U);

    //volatile 
    auto transformed = tswdft2d<float>((float*)img.data, WINDOW_DIMENSION, WINDOW_DIMENSION, img.rows, img.cols);

    cv::Mat visualization(visualizationRows, visualizationCols, CV_32FC1);
    cv::Mat amplitude(visualizationRows, visualizationCols, CV_32FC1);
    cv::Mat vertical(visualizationRows, visualizationCols, CV_32FC1);

    for (int y = 0; y < visualizationRows; ++y)
        for (int x = 0; x < visualizationCols; ++x)
        {
            const auto sourceOffset = y * visualizationCols + x;

            //float offsets[WINDOW_DIMENSION * WINDOW_DIMENSION - 1];
            //float amplitudes[WINDOW_DIMENSION * WINDOW_DIMENSION - 1];

            std::map<float, float> ordered;

            double v = 0;

            for (int j = 1; j < WINDOW_DIMENSION/* * WINDOW_DIMENSION*/; ++j)
            {
                if (j / WINDOW_DIMENSION > WINDOW_DIMENSION / 2 || j % WINDOW_DIMENSION > WINDOW_DIMENSION / 2)
                    continue;

                const auto amplitude = std::abs(transformed[sourceOffset * WINDOW_DIMENSION * WINDOW_DIMENSION + j]) / sqrtf(j);
                const auto freq = hypot(j / WINDOW_DIMENSION, j % WINDOW_DIMENSION);
                if (freq > 2)
                    ordered[freq] = std::max(ordered[freq], amplitude);

                if (j % WINDOW_DIMENSION == 0)
                    v += amplitude;
            }

            auto it = ordered.begin();
            auto freq = it->first;
            auto threshold = it->second;

            while (++it != ordered.end())
            {
                if (it->second > threshold)
                {
                    freq = it->first;
                    threshold = it->second;
                }
                //else if (it->second < threshold / 10.)
                //    break;
            }

            visualization.at<float>(y, x) = logf(freq + 1.);
            amplitude.at<float>(y, x) = logf(threshold + 1.);

            vertical.at<float>(y, x) = logf(v + 1.);
        }

    cv::normalize(visualization, visualization, 0, 1, cv::NORM_MINMAX);
    cv::normalize(amplitude, amplitude, 0, 1, cv::NORM_MINMAX);
    cv::normalize(vertical, vertical, 0, 1, cv::NORM_MINMAX);

    cv::imshow("func", func);

    cv::imshow("visualization", visualization);

    cv::imshow("amplitude", amplitude);

    cv::imshow("vertical", vertical);

    cv::Mat borderline(visualizationRows, visualizationCols, CV_8UC1);

    // border line
    for (int y = 0; y < visualizationRows - 1; ++y)
        for (int x = 0; x < visualizationCols; ++x)
        {
            const auto sourceOffset1 = y * visualizationCols + x;
            const auto sourceOffset2 = (y + 1) * visualizationCols + x;

            int freq1 = 0;
            int freq2 = 0;

            float threshold1 = 0;
            float threshold2 = 0;

            for (int j = 3; j <= WINDOW_DIMENSION / 2; ++j)
            {
                const auto amplitude1 = std::abs(transformed[sourceOffset1 * WINDOW_DIMENSION * WINDOW_DIMENSION + j]) / sqrt(j);
                const auto amplitude2 = std::abs(transformed[sourceOffset2 * WINDOW_DIMENSION * WINDOW_DIMENSION + j]) / sqrt(j);
                if (amplitude1 > threshold1)
                {
                    freq1 = j;
                    threshold1 = amplitude1;
                }
                if (amplitude2 > threshold2)
                {
                    freq2 = j;
                    threshold2 = amplitude2;
                }
            }
            //if (freq1 > 2 && freq1 >= ((freq2 * 3 / 5 - 1)) && freq1 <= ((freq2 * 3 / 5 + 1)))
            if (freq2 >= freq1 * 5 / 3 && freq2 <= freq1 * 5 / 2)
            {
                borderline.at<uchar>(y, x) = 255;
            }
        }

    cv::imshow("borderline", borderline);


    cv::setMouseCallback("func", CallBackFunc, &transformed);


    cv::waitKey(0);

    return 0;

}
