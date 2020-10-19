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

#include <random>

void calcGST(const cv::Mat& inputImg, cv::Mat& imgCoherencyOut, cv::Mat& imgOrientationOut, int w = 52)
{
    using namespace cv;

    Mat img;
    inputImg.convertTo(img, CV_32F);
    // GST components calculation (start)
    // J =  (J11 J12; J12 J22) - GST
    Mat imgDiffX, imgDiffY, imgDiffXY;
    Sobel(img, imgDiffX, CV_32F, 1, 0, 3);
    Sobel(img, imgDiffY, CV_32F, 0, 1, 3);
    multiply(imgDiffX, imgDiffY, imgDiffXY);
    Mat imgDiffXX, imgDiffYY;
    multiply(imgDiffX, imgDiffX, imgDiffXX);
    multiply(imgDiffY, imgDiffY, imgDiffYY);
    Mat J11, J22, J12;      // J11, J22 and J12 are GST components
    boxFilter(imgDiffXX, J11, CV_32F, Size(w, w));
    boxFilter(imgDiffYY, J22, CV_32F, Size(w, w));
    boxFilter(imgDiffXY, J12, CV_32F, Size(w, w));
    // GST components calculation (stop)
    // eigenvalue calculation (start)
    // lambda1 = 0.5*(J11 + J22 + sqrt((J11-J22)^2 + 4*J12^2))
    // lambda2 = 0.5*(J11 + J22 - sqrt((J11-J22)^2 + 4*J12^2))
    Mat tmp1, tmp2, tmp3, tmp4;
    tmp1 = J11 + J22;
    tmp2 = J11 - J22;
    multiply(tmp2, tmp2, tmp2);
    multiply(J12, J12, tmp3);
    sqrt(tmp2 + 4.0 * tmp3, tmp4);
    Mat lambda1, lambda2;
    lambda1 = tmp1 + tmp4;
    lambda1 = 0.5*lambda1;      // biggest eigenvalue
    lambda2 = tmp1 - tmp4;
    lambda2 = 0.5*lambda2;      // smallest eigenvalue
    // eigenvalue calculation (stop)
    // Coherency calculation (start)
    // Coherency = (lambda1 - lambda2)/(lambda1 + lambda2)) - measure of anisotropism
    // Coherency is anisotropy degree (consistency of local orientation)
    divide(lambda1 - lambda2, lambda1 + lambda2, imgCoherencyOut);
    // Coherency calculation (stop)
    // orientation angle calculation (start)
    // tan(2*Alpha) = 2*J12/(J22 - J11)
    // Alpha = 0.5 atan2(2*J12/(J22 - J11))
    phase(J22 - J11, 2.0*J12, imgOrientationOut, false);
    imgOrientationOut = 0.5*imgOrientationOut;
    // orientation angle calculation (stop)
}


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
        for (int j = 1; j < WINDOW_DIMENSION * WINDOW_DIMENSION; ++j)
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

//////////////////////////////////////////////////////////////////////////////

bool polynomial_curve_fit(const std::vector<cv::Point2d>& key_point, int n, cv::Mat& A)
{
    //Number of key points
    int N = key_point.size();

    //Construct matrix X
    cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int j = 0; j < n + 1; j++)
        {
            for (int k = 0; k < N; k++)
            {
                X.at<double>(i, j) = X.at<double>(i, j) +
                    std::pow(key_point[k].x, i + j);
            }
        }
    }

    //Construct matrix Y
    cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int k = 0; k < N; k++)
        {
            Y.at<double>(i, 0) = Y.at<double>(i, 0) +
                std::pow(key_point[k].x, i) * key_point[k].y;
        }
    }

    A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    //Solve matrix A
    cv::solve(X, Y, A, cv::DECOMP_LU);
    return true;
}


// Generate a uniform distribution of the number between[0, 1]
double uniformRandom(void)
{
    return (double)rand() / (double)RAND_MAX;
}

/*
// Fit the line according to the point set ax + by + c = 0, res is the residual
void calcLinePara(const std::vector<cv::Point2d>& pts, double &a, double &b, double &c, double &res)
{
    res = 0;
    cv::Vec4f line;
    std::vector<cv::Point2f> ptsF;
    for (unsigned int i = 0; i < pts.size(); i++)
        ptsF.push_back(pts[i]);

    cv::fitLine(ptsF, line, cv::DIST_L2, 0, 1e-2, 1e-2);
    a = line[1];
    b = -line[0];
    c = line[0] * line[3] - line[1] * line[2];

    for (unsigned int i = 0; i < pts.size(); i++)
    {
        double resid_ = fabs(pts[i].x * a + pts[i].y * b + c);
        res += resid_;
    }
    res /= pts.size();
}
*/

// Get a straight line fitting sample, that is, randomly select 2 points on the line sampling point set
#if 0
bool getSample(const std::vector<int>& set, std::vector<int> &sset, int num)
{
    if (set.size() <= num)
        return false;

    std::map<int, int> displaced;

    sset.resize(num);

    std::default_random_engine dre;
    std::uniform_int_distribution<int> di(0, set.size() - 1);

    for (int i = 0; i < num; ++i)
    {
        int idx = di(dre);
        int v;
        if (idx == i)
            v = i;
        else
        {
            auto it = displaced.find(idx);
            if (it != displaced.end())
            {
                v = it->second;
                it->second = i;
            }
            else
            {
                v = idx;
                displaced[idx] = i;
            }
        }

        sset[i] = v;
    }

    return true;

    //int i[2];
    //sset.resize(n);
    //if (set.size() > 2)
    //{
    //    do
    //    {
    //        for (int n = 0; n < 2; n++)
    //            i[n] = int(uniformRandom() * (set.size() - 1));
    //    } while (!(i[1] != i[0]));
    //    for (int n = 0; n < 2; n++)
    //    {
    //        sset.push_back(i[n]);
    //    }
    //}
    //else
    //{
    //    return false;
    //}
    //return true;
}
#endif

//The position of two random points in the line sample cannot be too close
bool verifyComposition(const std::vector<cv::Point2d>& pts)
{
    //cv::Point2d pt1 = pts[0];
    //cv::Point2d pt2 = pts[1];
    //if (abs(pt1.x - pt2.x) < 5 && abs(pt1.y - pt2.y) < 5)
    //    return false;

    for (int i = 1; i < pts.size(); ++i)
        for (int j = 0; j < i; ++j)
        {
            cv::Point2d pt1 = pts[j];
            cv::Point2d pt2 = pts[i];
            if (abs(pt1.x - pt2.x) < 5 && abs(pt1.y - pt2.y) < 5)
                return false;
        }

    return true;
}


//RANSAC straight line fitting
void fitLineRANSAC(const std::vector<cv::Point2d>& ptSet, 
    //double &a, double &b, double &c, 
    cv::Mat& a, int n_samples,
    std::vector<bool> &inlierFlag)
{
    //double residual_error = 2.99; // inner point threshold
    const double residual_error = 10; // inner point threshold

    bool stop_loop = false;
    int maximum = 0; //maximum number of points

    //final inner point identifier and its residual
    inlierFlag = std::vector<bool>(ptSet.size(), false);
    std::vector<double> resids_;// (ptSet.size(), 3);
    int sample_count = 0;
    int N = 100000;

    //double res = 0;

    // RANSAC
    srand((unsigned int)time(NULL)); //Set random number seed
    std::vector<int> ptsID;
    for (unsigned int i = 0; i < ptSet.size(); i++)
        ptsID.push_back(i);

    //enum { n_samples  = 8 };

    std::vector<int> ptss(n_samples);

    std::default_random_engine dre;

    while (N > sample_count && !stop_loop)
    {
        cv::Mat res;

        std::vector<bool> inlierstemp;
        std::vector<double> residualstemp;
        //std::vector<int> ptss;
        int inlier_count = 0;

        //random sampling - n_samples points
        for (int j = 0; j < n_samples; ++j)
            ptss[j] = j;

        std::map<int, int> displaced;

        // Fisher-Yates shuffle Algorithm
        for (int j = 0; j < n_samples; ++j)
        {
            std::uniform_int_distribution<int> di(j, ptsID.size() - 1);
            int idx = di(dre);

            if (idx != j)
            {
                int& to_exchange = (idx < n_samples) ? ptss[idx] : displaced.try_emplace(idx, idx).first->second;
                std::swap(ptss[j], to_exchange);
            }
        }


        //if (!getSample(ptsID, ptss, 3))
        //{
        //    stop_loop = true;
        //    continue;
        //}

        std::vector<cv::Point2d> pt_sam;
        for (int i = 0; i < n_samples; ++i)
            pt_sam.push_back(ptSet[ptss[i]]);

        //pt_sam.push_back(ptSet[ptss[0]]);
        //pt_sam.push_back(ptSet[ptss[1]]);

        if (!verifyComposition(pt_sam))
        {
            ++sample_count;
            continue;
        }

        // Calculate the line equation
            //calcLinePara(pt_sam, a, b, c, res);
        polynomial_curve_fit(pt_sam, n_samples - 1, res);
        //Inside point test
        for (unsigned int i = 0; i < ptSet.size(); i++)
        {
            cv::Point2d pt = ptSet[i];
            auto x = ptSet[i].x;
            //double resid_ = fabs(pt.x * a + pt.y * b + c);

            //double y = res.at<double>(0, 0) + res.at<double>(1, 0) * x +
            //    res.at<double>(2, 0)*std::pow(x, 2) + res.at<double>(3, 0)*std::pow(x, 3);

            double y = res.at<double>(0, 0) + res.at<double>(1, 0) * x;
            for (int i = 2; i < n_samples; ++i)
                y += res.at<double>(i, 0) * std::pow(x, i);

            double resid_ = fabs(ptSet[i].y - y);

            residualstemp.push_back(resid_);
            inlierstemp.push_back(false);
            if (resid_ < residual_error)
            {
                ++inlier_count;
                inlierstemp[i] = true;
            }
        }
        // find the best fit straight line
        if (inlier_count >= maximum)
        {
            maximum = inlier_count;
            resids_ = residualstemp;
            inlierFlag = inlierstemp;
        }
        // Update the number of RANSAC iterations, as well as the probability of interior points
        if (inlier_count == 0)
        {
            N = 500;
        }
        else
        {
            double epsilon = 1.0 - double(inlier_count) / (double)ptSet.size(); // wild value point scale
            double p = 0.99; //the probability of having 1 good sample in all samples
            double s = 2.0;
            N = int(log(1.0 - p) / log(1.0 - pow((1.0 - epsilon), s)));
        }
        ++sample_count;
    }

    // Use all the interior points to re - fit the line
    std::vector<cv::Point2d> pset;
    for (unsigned int i = 0; i < ptSet.size(); i++)
    {
        if (inlierFlag[i])
            pset.push_back(ptSet[i]);
    }

    //calcLinePara(pset, a, b, c, res);
    polynomial_curve_fit(pset, n_samples - 1, a);
}

//////////////////////////////////////////////////////////////////////////////

double CalcPoly(const cv::Mat& X, double x)
{
    double result = X.at<double>(0, 0);
    double v = 1.;
    for (int i = 1; i < X.rows; ++i)
    {
        v *= x;
        result += X.at<double>(i, 0) * v;
    }
    return result;
}

void fitLineRANSAC2(const std::vector<cv::Point2d>& vals, cv::Mat& a, int n_samples, std::vector<bool> &inlierFlag, double noise_sigma = 5.)
{
    //int n_data = vals.size();
    int N = 100000;	//iterations 
    double T = 3 * noise_sigma;   // residual threshold

    //int n_sample = 3;

    //int max_cnt = 0;

    double max_weight = 0.;

    cv::Mat best_model(n_samples, 1, CV_64FC1);

    std::default_random_engine dre;

    std::vector<int> k(n_samples);

    for (int n = 0; n < N; n++)
    {
        //random sampling - n_samples points
        for (int j = 0; j < n_samples; ++j)
            k[j] = j;

        std::map<int, int> displaced;

        // Fisher-Yates shuffle Algorithm
        for (int j = 0; j < n_samples; ++j)
        {
            std::uniform_int_distribution<int> di(j, vals.size() - 1);
            int idx = di(dre);

            if (idx != j)
            {
                int& to_exchange = (idx < n_samples) ? k[idx] : displaced.try_emplace(idx, idx).first->second;
                std::swap(k[j], to_exchange);
            }
        }

        //printf("random sample : %d %d %d\n", k[0], k[1], k[2]);

        //model estimation
        cv::Mat AA(n_samples, n_samples, CV_64FC1);
        cv::Mat BB(n_samples, 1, CV_64FC1);
        for (int i = 0; i < n_samples; i++)
        {
            AA.at<double>(i, 0) = 1.;
            double v = 1.;
            for (int j = 1; j < n_samples; ++j)
            {
                v *= vals[k[i]].x;
                AA.at<double>(i, j) = v;
            }

            BB.at<double>(i, 0) = vals[k[i]].y;
        }

        cv::Mat AA_pinv(n_samples, n_samples, CV_64FC1);
        invert(AA, AA_pinv, cv::DECOMP_SVD);

        cv::Mat X = AA_pinv * BB;

        //evaluation 
        //int cnt = 0;
        double weight = 0.;
        for (const auto& v : vals)
        {
            double data = std::abs(v.y - CalcPoly(X, v.x));

            weight += exp(-data * data / (2 * noise_sigma * noise_sigma));

            //if (data < T)
            //{
            //    cnt++;
            //}
        }

        //if (cnt > max_cnt)
        if (weight > max_weight)
        {
            best_model = X;
            max_weight = weight;
        }
    }

    //------------------------------------------------------------------- optional LS fitting 
    inlierFlag = std::vector<bool>(vals.size(), false);
    std::vector<int> vec_index;
    for (int i = 0; i < vals.size(); i++)
    {
        const auto& v = vals[i];
        double data = std::abs(v.y - CalcPoly(best_model, v.x));
        if (data < T)
        {
            inlierFlag[i] = true;
            vec_index.push_back(i);
        }
    }

    cv::Mat A2(vec_index.size(), n_samples, CV_64FC1);
    cv::Mat B2(vec_index.size(), 1, CV_64FC1);

    for (int i = 0; i < vec_index.size(); i++)
    {
        A2.at<double>(i, 0) = 1.;
        double v = 1.;
        for (int j = 1; j < n_samples; ++j)
        {
            v *= vals[vec_index[i]].x;
            A2.at<double>(i, j) = v;
        }


        B2.at<double>(i, 0) = vals[vec_index[i]].y;
    }

    cv::Mat A2_pinv(n_samples, vec_index.size(), CV_64FC1);
    invert(A2, A2_pinv, cv::DECOMP_SVD);

    a = A2_pinv * B2;

    //return X;
}

//////////////////////////////////////////////////////////////////////////////

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

    cv::Mat imgCoherency, imgOrientation;
    calcGST(funcFloat, imgCoherency, imgOrientation);

    //volatile 
    auto transformed = tswdft2d<float>((float*)img.data, WINDOW_DIMENSION, WINDOW_DIMENSION, img.rows, img.cols);

    cv::Mat visualization(visualizationRows, visualizationCols, CV_32FC1);
    cv::Mat amplitude(visualizationRows, visualizationCols, CV_32FC1);

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

            //vertical.at<float>(y, x) = logf(v + 1.);
        }

    cv::Mat vertical(visualizationRows, visualizationCols, CV_32FC1);
    for (int y = 0; y < visualizationRows; ++y)
        for (int x = 0; x < visualizationCols; ++x)
        {
            const auto sourceOffset = y * visualizationCols + x;

            //float offsets[WINDOW_DIMENSION * WINDOW_DIMENSION - 1];
            //float amplitudes[WINDOW_DIMENSION * WINDOW_DIMENSION - 1];

            //std::map<float, float> ordered;

            double v = 0;

            for (int j = 1; j < WINDOW_DIMENSION* WINDOW_DIMENSION; ++j)
            {
                if (j / WINDOW_DIMENSION > WINDOW_DIMENSION / 2 || j % WINDOW_DIMENSION > WINDOW_DIMENSION / 2)
                    continue;

                const auto amplitude = std::abs(transformed[sourceOffset * WINDOW_DIMENSION * WINDOW_DIMENSION + j]);// / sqrtf(j);
                //const auto freq = hypot(j / WINDOW_DIMENSION, j % WINDOW_DIMENSION);
                //if (freq > 2)
                //    ordered[freq] = std::max(ordered[freq], amplitude);

                if (j % WINDOW_DIMENSION == 0)
                    v += amplitude;
            }

            vertical.at<float>(y, x) = logf(v + 1.);
        }




    cv::normalize(visualization, visualization, 0, 1, cv::NORM_MINMAX);
    cv::normalize(amplitude, amplitude, 0, 1, cv::NORM_MINMAX);
    cv::normalize(vertical, vertical, 0, 1, cv::NORM_MINMAX);

    cv::imshow("func", func);

    cv::imshow("visualization", visualization);

    cv::imshow("amplitude", amplitude);

    cv::imshow("vertical", vertical);

    cv::Mat borderline(visualizationRows, visualizationCols, CV_8UC1, cv::Scalar(0));

    // border line
    std::vector<cv::Point2d> ptSet;


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
            if (freq2 > freq1 && freq2 >= freq1 * 5 / 3 && freq2 <= freq1 * 5 / 2)
            {
                const auto coherency = imgCoherency.at<float>(y + WINDOW_DIMENSION / 2, x + WINDOW_DIMENSION / 2);
                if (coherency > 0.5)
                    //borderline.at<uchar>(y, x) = 255;
                    ptSet.push_back(cv::Point2d(x, y));
            }
        }

    //double A, B, C;

    enum { n_samples  = 8 };
    cv::Mat A;
    std::vector<bool> inliers;
    fitLineRANSAC2(ptSet, A, n_samples, //A, B, C, 
        inliers);

#if 0
    {
        std::vector<cv::Point2d> ptSet2;
        for (unsigned int i = 0; i < ptSet.size(); ++i) {
            if (inliers[i])
                ptSet2.push_back(ptSet[i]);
        }

        std::vector<bool> inliers2;
        fitLineRANSAC2(ptSet2, A, n_samples, //A, B, C, 
            inliers2);

        std::swap(ptSet, ptSet2);
        std::swap(inliers, inliers2);
    }
#endif

    for (unsigned int i = 0; i < ptSet.size(); ++i) {
        if (inliers[i])
            borderline.at<uchar>(ptSet[i].y, ptSet[i].x) = 255;
    }

    cv::imshow("borderline", borderline);

    std::vector<cv::Point> points_fitted;
    for (int x = 0; x < visualizationCols; x++)
    {
        //double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x +
        //    A.at<double>(2, 0)*std::pow(x, 2) + A.at<double>(3, 0)*std::pow(x, 3);

        double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x;
        for (int i = 2; i < n_samples; ++i)
            y += A.at<double>(i, 0) * std::pow(x, i);

        points_fitted.push_back(cv::Point(x + WINDOW_DIMENSION / 2, y + WINDOW_DIMENSION / 2));
    }

    cv::polylines(func, points_fitted, false, cv::Scalar(0, 255, 255), 1, 8, 0);
    cv::imshow("image", func);


    //imgCoherency *= 10;
    //cv::exp(imgCoherency, imgCoherency);

    cv::normalize(imgCoherency, imgCoherency, 0, 1, cv::NORM_MINMAX);
    cv::normalize(imgOrientation, imgOrientation, 0, 1, cv::NORM_MINMAX);

    cv::imshow("imgCoherency", imgCoherency);
    cv::imshow("imgOrientation", imgOrientation);


    cv::setMouseCallback("func", CallBackFunc, &transformed);


    cv::waitKey(0);

    return 0;

}
