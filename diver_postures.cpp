// In this, first get the brightest frame. Then subtract the background from the foreground.
// After this, perform some morpholigical operation to get the proper masking. Replace the pixels
// from the best frame to the main frame in relation to rows and cols of this masking. Repeat this process
// for another set of frame until the end of video and get our desired foreground mosaic.

#include <iostream>
#include "opencv2/bgsegm.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::bgsegm;

Mat res;
Mat next_frame;
Mat best_frame;
Mat vidframe;
Mat fgMask;
Mat element;
Mat main_frame;
Mat main_gray;
Mat updated;
VideoCapture vid("Video/IMG_1794__2175.m4v");
int brightness = 0;
int total = 0;
int num = 0;
int avg = 0;
int erosion_type = 2;
int erosion_size = 2;
int num_frames = 0;

int main(int argc, char** argv) {
    VideoCapture vidmain("Video/IMG_1800__2175.m4v");
    if (!vidmain.read(main_frame)) {
        return 0;
    }
    vidmain.release();
    cvtColor(main_frame, main_gray, CV_BGR2GRAY);
    updated = Mat::zeros(main_gray.rows, main_gray.cols, CV_64F);
    VideoCapture vid("Video/IMG_1800__2175.m4v");
    Ptr<BackgroundSubtractor> pBgSub = createBackgroundSubtractorMOG(200, 5, 0.0001);
    num_frames = vid.get(CV_CAP_PROP_FRAME_COUNT);
    for (int index = 0; index <=100; index ++) {
        if (!vid.read(vidframe)) {
            break;
        }
    }
    int no_frame = 0;
    for (int index = 1; index < 10; index++) {
        if (no_frame == 1) {
            break;
        }
        brightness = 0;
        for (int iter = 1; iter <=100; iter++) {
            total = 0;
            avg = 0;
            if (!vid.read(vidframe)) {
                no_frame = 1;
                break;
            }
            vidframe.copyTo(next_frame);
            // get the brightest frame from the average
            for (int row_no = 0; row_no < vidframe.rows; row_no++) {
                for (int col_no = 0; col_no < vidframe.cols; col_no++) {
                    Vec3b color = vidframe.at<Vec3b>(Point(row_no, col_no));
                    total = total + (int)color[0] + (int)color[1] + (int)color[2];
                }
            }
            num = vidframe.rows * vidframe.cols * 3;
            avg = total / num;
            if ( avg > brightness ) {
                brightness = avg;
                vidframe.copyTo(best_frame);
            }
            if (iter != 6) {
                continue;
            }
            // separate the background from the foreground and get the masking
            pBgSub->apply(best_frame, fgMask);
            element = getStructuringElement( erosion_type, Size(2*erosion_size + 1, 2*erosion_size + 1));
            erode(fgMask, fgMask, element);
            
            erosion_size = 7;
            element = getStructuringElement( erosion_type, Size(2*erosion_size + 1, 2*erosion_size + 1));
            // perform morphological operations
            dilate(fgMask, fgMask, element);
            dilate(fgMask, fgMask, element);
            dilate(fgMask, fgMask, element);
            dilate(fgMask, fgMask, element);
            dilate(fgMask, fgMask, element);
            dilate(fgMask, fgMask, element);
            // replace the pixel from the best frame to the main frame and keep track of it
            for (int row_no = 0; row_no < main_frame.rows; row_no++) {
                for (int col_no = 0; col_no < main_frame.cols; col_no++) {
                    if (fgMask.at<uchar>(row_no, col_no) != 0) {
                        if (updated.at<double>(row_no, col_no) != 0) {
                            continue;
                        }
                        main_frame.at<Vec3b>(row_no, col_no) = best_frame.at<Vec3b>(row_no, col_no);
                        updated.at<double>(row_no, col_no) = 255;
                        
                    }
                }
            }
//            imshow("New", fgMask);
//            imshow("Video", main_frame);
            if (waitKey(30) == 27) {
                break;
            }
        }
    }
    vid.release();
    cout << "Done.\n";
    imshow("Colab", main_frame);
    waitKey(0);
    return 0;
}
        
