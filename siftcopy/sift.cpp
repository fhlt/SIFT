#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <cmath>

using namespace std;
using namespace cv;

// 下述两个变量都是针对DoG而言的
#define NUM_OCTAVES 4             // 4层金子塔
#define NUM_SCALES 5              // 5层极值点

#define ANTIALIAS_SIGMA         0.5
#define KERNEL_SIZE             5
#define CONTRAST_THRESHOLD      0.03
#define INIT_SIGMA              1.4142135	//sqrt(2)
#define PREBLUR_SIGMA           1.0
#define R_CURVATURE             10.0

typedef std::vector<Mat> Array;
typedef std::vector<Array> TwoDArray;
TwoDArray ScaleSpace, DoG, DoG_Keypts;

void init_array(int num_octaves, int num_scales)
{
	for (int oct = 0;oct < num_octaves;oct++)
	{
		ScaleSpace.push_back(Array(num_scales + 3));
		DoG.push_back(Array(num_scales + 2));
		DoG_Keypts.push_back(Array(num_scales));
	}
}

void CreateScaleSpace(Mat source)
{
	double sigma;
	Mat src_antialiased, up, down;
	Size ksize(KERNEL_SIZE, KERNEL_SIZE);
	GaussianBlur(source, src_antialiased, ksize, ANTIALIAS_SIGMA);  // 对原图进行高斯滤波
//	imshow("src_antialiased", src_antialiased);  
	pyrUp(src_antialiased, up);    // 上采样
//	imshow("up", up);
	up.copyTo(ScaleSpace[0][0]);

	GaussianBlur(ScaleSpace[0][0], ScaleSpace[0][0], ksize, PREBLUR_SIGMA);
	for (int oct = 0;oct < NUM_OCTAVES; oct++)
	{
		sigma = INIT_SIGMA;
		for (int sc = 0; sc < NUM_SCALES + 2; sc++)
		{
			sigma = sigma * pow(2.0, sc / 2.0);
			GaussianBlur(ScaleSpace[oct][sc], ScaleSpace[oct][sc + 1], ksize, sigma);
			DoG[oct][sc] = ScaleSpace[oct][sc] - ScaleSpace[oct][sc + 1];
		}

		// 为下一层做初始化
		if (oct < NUM_OCTAVES - 1)
		{
			pyrDown(ScaleSpace[oct][0], down);
			down.copyTo(ScaleSpace[oct + 1][0]);
		}
	}
}

void DoGExtrema()
{
	Mat local_maxima, local_minima, extrema, current, top, down;
	for (int oct = 0;oct < NUM_OCTAVES;oct++)
	{
		for (int sc = 0; sc < NUM_SCALES;sc++)
		{
			DoG_Keypts[oct][sc] = Mat::zeros(DoG[oct][sc].size(), DoG[oct][sc].type());
			top = DoG[oct][sc];
			current = DoG[oct][sc + 1];
			down = DoG[oct][sc + 2];
			int sx = current.rows;
			int sy = current.cols;
			// 极值点检测，搜索每个点的26邻域，若该点为局部极值点，则保存为候选关键点
			// 极值包括极大值和极小值
			// local_maxima size（sx-2, sy-2）
			// current 8个点
			local_maxima = (current(Range(1, sx - 1), Range(1, sy - 1)) > current(Range(0, sx - 2), Range(0, sy - 2))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > current(Range(0, sx - 2), Range(1, sy - 1))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > current(Range(0, sx - 2), Range(2, sy))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > current(Range(1, sx - 1), Range(0, sy - 2))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > current(Range(1, sx - 1), Range(2, sy))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > current(Range(2, sx), Range(0, sy - 2))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > current(Range(2, sx), Range(1, sy - 1))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > current(Range(2, sx), Range(2, sy)));
			// top 9个点
			local_maxima = local_maxima & (current(Range(1, sx - 1), Range(1, sy - 1)) > top(Range(0, sx - 2), Range(0, sy - 2))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > top(Range(0, sx - 2), Range(1, sy - 1))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > top(Range(0, sx - 2), Range(2, sy))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > top(Range(1, sx - 1), Range(0, sy - 2))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > top(Range(1, sx - 1), Range(1, sy - 1))) &  // same pixel in top
				(current(Range(1, sx - 1), Range(1, sy - 1)) > top(Range(1, sx - 1), Range(2, sy))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > top(Range(2, sx), Range(0, sy - 2))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > top(Range(2, sx), Range(1, sy - 1))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > top(Range(2, sx), Range(2, sy)));
			// down 9个点
			local_maxima = local_maxima & (current(Range(1, sx - 1), Range(1, sy - 1)) > down(Range(0, sx - 2), Range(0, sy - 2))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > down(Range(0, sx - 2), Range(1, sy - 1))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > down(Range(0, sx - 2), Range(2, sy))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > down(Range(1, sx - 1), Range(0, sy - 2))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > down(Range(1, sx - 1), Range(1, sy - 1))) &  // same pixel in top
				(current(Range(1, sx - 1), Range(1, sy - 1)) > down(Range(1, sx - 1), Range(2, sy))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > down(Range(2, sx), Range(0, sy - 2))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > down(Range(2, sx), Range(1, sy - 1))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) > down(Range(2, sx), Range(2, sy)));

			local_minima = (current(Range(1, sx - 1), Range(1, sy - 1)) < current(Range(0, sx - 2), Range(0, sy - 2))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < current(Range(0, sx - 2), Range(1, sy - 1))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < current(Range(0, sx - 2), Range(2, sy))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < current(Range(1, sx - 1), Range(0, sy - 2))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < current(Range(1, sx - 1), Range(2, sy))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < current(Range(2, sx), Range(0, sy - 2))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < current(Range(2, sx), Range(1, sy - 1))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < current(Range(2, sx), Range(2, sy)));
			local_minima = local_maxima & (current(Range(1, sx - 1), Range(1, sy - 1)) < top(Range(0, sx - 2), Range(0, sy - 2))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < top(Range(0, sx - 2), Range(1, sy - 1))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < top(Range(0, sx - 2), Range(2, sy))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < top(Range(1, sx - 1), Range(0, sy - 2))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < top(Range(1, sx - 1), Range(1, sy - 1))) &  // same pixel in top
				(current(Range(1, sx - 1), Range(1, sy - 1)) < top(Range(1, sx - 1), Range(2, sy))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < top(Range(2, sx), Range(0, sy - 2))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < top(Range(2, sx), Range(1, sy - 1))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < top(Range(2, sx), Range(2, sy)));
			local_minima = local_maxima & (current(Range(1, sx - 1), Range(1, sy - 1)) < down(Range(0, sx - 2), Range(0, sy - 2))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < down(Range(0, sx - 2), Range(1, sy - 1))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < down(Range(0, sx - 2), Range(2, sy))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < down(Range(1, sx - 1), Range(0, sy - 2))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < down(Range(1, sx - 1), Range(1, sy - 1))) &  // same pixel in top
				(current(Range(1, sx - 1), Range(1, sy - 1)) < down(Range(1, sx - 1), Range(2, sy))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < down(Range(2, sx), Range(0, sy - 2))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < down(Range(2, sx), Range(1, sy - 1))) &
				(current(Range(1, sx - 1), Range(1, sy - 1)) < down(Range(2, sx), Range(2, sy)));
			extrema = local_maxima | local_minima;   // 极大值或者极小值均可
			int count = 0;
			
			for (int m = 0;m < extrema.rows;m++)
			{
				for (int n=0;n < extrema.cols;n++)
				{
					if (extrema.at<uchar>(m,n))
						count++;
				}
			}
			
			cout << "count :" << count << endl;
//			imshow("extema", extrema);
			extrema.copyTo(DoG_Keypts[oct][sc](Range(1, DoG_Keypts[oct][sc].rows - 1), Range(1, DoG_Keypts[oct][sc].cols - 1)));
		}
	}
}

void display_images(int oct, int sc, int mode)
{
	stringstream ss_oct, ss_sc;
	ss_oct << oct; ss_sc << sc;
	switch (mode)
	{
	case 1:        // Display the Scale Space images
	{
		string wSS = "SS Octave " + ss_oct.str() + " Scale " + ss_sc.str();
		imshow(wSS, ScaleSpace[oct - 1][sc - 1]);
		break;
	}
	case 2:        // Display the DoG images
	{
		string wDoG = "DoG Octave " + ss_oct.str() + " Scale " + ss_sc.str();
		imshow(wDoG, DoG[oct - 1][sc - 1]);
		break;
	}
	case 3:       // Display DoG Keypoints
	{
		string wDoGKP = "DoG Keypts Octave " + ss_oct.str() + " Scale " + ss_sc.str();
		Mat ResultImage = ScaleSpace[oct - 1][sc - 1];
		Mat features = DoG_Keypts[oct - 1][sc - 1];
		for (int i = 0;i < features.rows;i++)
		{
			for (int j = 0;j <features.cols;j++)
			{
				if (features.at<uchar>(i, j))
				{
					circle(ResultImage, Point(i, j), 5, Scalar(0, 0, 255), 2, 8, 0);
				}
			}
		}
		imshow(wDoGKP, ResultImage);
		break;
	}
	}
}

// 去除低对比度的点及边缘上的点
void FilterDoGExtrema()
{
	Mat locs;
	int x, y, fxx, fyy, fxy;
	float trace, det, curvature;
	float curv_threshold = pow((R_CURVATURE + 1), 2) / R_CURVATURE;
	for (int oct = 0;oct < NUM_OCTAVES;oct++)
	{
		for (int sc = 0;sc < NUM_SCALES;sc++)
		{
			findNonZero(DoG_Keypts[oct][sc], locs);
			int num_keypts = locs.cols;
			Mat_<uchar> current = DoG[oct][sc + 1];
			for (int k = 0;k < num_keypts;k++)
			{
				x = locs.at<int>(k, 0);
				y = locs.at<int>(k, 1);
				if (abs(current(x, y)) < CONTRAST_THRESHOLD)
				{
					DoG_Keypts[oct][sc].at<uchar>(x, y) = 0;
				}
				else
				{
					fxx = current(x - 1, y) + current(x + 1, y) - 2 * current(x, y); 
					fyy = current(x, y - 1) + current(x, y + 1) - 2 * current(x, y);
					fxy = current(x - 1, y - 1) + current(x + 1, y + 1) - current(x - 1, y + 1) - current(x + 1, y - 1);
					trace = (float)(fxx + fyy);
					det = (fxx*fyy) - (fxy*fxy);
					curvature = (float)(trace*trace / det);
					//去除边缘上的点
					if (det < 0 || curvature > curv_threshold)
					{
						DoG_Keypts[oct][sc].at<uchar>(x, y) = 0;
					}
				}
			}
		}
	}
}

int main()
{
	Mat source;
	source = imread("wall.jpg", 0);
	init_array(NUM_OCTAVES, NUM_SCALES);
	CreateScaleSpace(source);
	DoGExtrema();
	FilterDoGExtrema();
	// 去除不稳定的关键点
	display_images(2, 1, 3);
	waitKey(0);
	return 0;
}