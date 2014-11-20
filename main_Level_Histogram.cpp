/// @file main.cpp
/// @Synopsis  
/// @author Aditya Singh <aditya.singh@research.iiit.ac.in>
/// @version 1.0
/// @date 2014-11-16


#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <opencv2/features2d/features2d.hpp>
#include <ctype.h>
#include <unistd.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <list>
#include <string>
#include <cstring>
#include <queue>
#include <utility>
using namespace std;
using namespace cv;


struct comparePQ{
    bool operator()(pair<int,double>p1,pair<int,double>p2){
        return p1.second<p2.second;
    }
};

vector<Mat> svcVector;
vector<Mat> svcHistogram;
list<Mat>epiQ;
priority_queue<pair<int,double>,vector<pair<int,double> >,comparePQ>histScore;
bool filter_I_selected[3000000];
int numEpiFrames;

void frameDetails(Mat &frame){
    int rows = frame.rows;
    int cols = frame.cols;
    cout<<"Episode: "<<cols<<" "<<rows<<endl;
}


/// @Synopsis  to check the performance of sift. Time, space of indexing. Time of feature generation.
void testingSIFT(Mat img){
    Mat input;
    cvtColor(img,input,CV_RGB2GRAY);
    Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
        std::vector<cv::KeyPoint> keypoints;
        cout<<input.channels()<<" "<<input.type()<<endl;
        detector->detect(input, keypoints);
                cv::Mat output;
                cv::drawKeypoints(input, keypoints, output);
                imshow("outputSIFT",output);
                waitKey(0);
    return;
}


bool LoadSVC(char *svcName){
    VideoCapture svc;
    svc.open(svcName);
    if(!svc.isOpened())
        return 0;

    Mat frame;

    while(true){
        svc>>frame;
        if(frame.empty())
            break;
        svcVector.push_back(frame);
   //     testingSIFT(frame);
        imshow("SVC", frame);
        waitKey(10);
    }
    cout<<svcVector.size()<<endl;
    return 1;
}
/*
   float frameSub(Mat &frame, int index){
   int Grows = gifVector[index].rows;
   int Gcols = gifVector[index].cols;
   int Erows = frame.rows;
   int Ecols = frame.cols;

// debugging
int maxval=-10000,minval=100000;
int maxval2=-10000,minval2=100000;
int ip;


Mat res,sub;
float cumVal=0;
int l = gifVector.size();  
if(Grows*Gcols < Erows*Ecols){
resize(frame,res,Size(Gcols,Grows),0,0,INTER_LINEAR);
absdiff(res,gifVector[index],sub);
int  Rrows = sub.rows,Rcols=sub.cols;

for(int i=0;i<Rrows;i++){
for(int j=0;j<Rcols;j++){
//      int f1 = res.at<Vec3b>(i,j)[0]+res.at<Vec3b>(i,j)[1]+res.at<Vec3b>(i,j)[2];
//        int f2 = gifVector[index].at<Vec3b>(i,j)[0]+gifVector[index].at<Vec3b>(i,j)[1]+gifVector[index].at<Vec3b>(i,j)[2];
//        cumVal += abs(f1-f2);
cumVal+= sub.at<Vec3b>(i,j)[0]+sub.at<Vec3b>(i,j)[1]+sub.at<Vec3b>(i,j)[2];

}
}
}
else{
if(Grows*Gcols > Erows*Ecols){
resize(gifVector[index],res,Size(Ecols,Erows),0,0,INTER_LINEAR);
absdiff(frame,res,sub);
int  Rrows = sub.rows,Rcols=sub.cols;
for(int i=0;i<Rrows;i++){
for(int j=0;j<Rcols;j++){
//     int f1 = res.at<Vec3b>(i,j)[0]+res.at<Vec3b>(i,j)[1]+res.at<Vec3b>(i,j)[2];
//    int f2 = frame.at<Vec3b>(i,j)[0]+frame.at<Vec3b>(i,j)[1]+frame.at<Vec3b>(i,j)[2];
//    cumVal += abs(f1-f2);
cumVal+= sub.at<Vec3b>(i,j)[0]+sub.at<Vec3b>(i,j)[1]+sub.at<Vec3b>(i,j)[2];
}
}
}
}
cumVal = cumVal/(Norm*float(res.rows*res.cols));
return cumVal; 
}

float CompareSEQS(){
int l = gifVector.size();
float compVal = 0;
for(int i=0;i<l;i++){
Mat eFrame = epiQ.front();
compVal+= frameSub(eFrame,i); 
epiQ.push_back(eFrame);
epiQ.pop_front();
}
epiQ.pop_front();
return compVal;
}
*/

void calcHistogram(Mat& src,Mat &b_hist,Mat &g_hist,Mat &r_hist){

    vector<Mat> bgr_planes;
    split( src, bgr_planes);
    /// Establish the number of bins
    int histSize = 256;

    //    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, false);
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, false);
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, false);

    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    float Norm = src.rows*src.cols;
    for(int i=0;i<256;i++){
        b_hist.at<float>(i,0)/=Norm;
        g_hist.at<float>(i,0)/=Norm;
        r_hist.at<float>(i,0)/=Norm;
    }

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    /*
       normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
       normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
       normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

       for( int i = 1; i < histSize; i++ )
       {
       line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
       Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
       Scalar( 255, 0, 0), 2, 8, 0  );
       line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
       Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
       Scalar( 0, 255, 0), 2, 8, 0  );
       line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
       Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
       Scalar( 0, 0, 255), 2, 8, 0  );
       }

       namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
       imshow("calcHist Demo", histImage );

       waitKey(0);
       */


}

void epiFilter_I_Processing(char *epiName){
    VideoCapture episode;
    episode.open(epiName);
    if(!episode.isOpened()){
        cout<<"Couldn't load the Videos\n";
        return;
    }
    int frame_num=0;
    Mat frame;
    fstream f;
    //1 Frame Per 10 Frames
    f.open("sample_1FP10F.txt",fstream::out);
    while(true){
        episode>>frame;
        if(frame.empty()){
            break;
        }
        frame_num++;
        if(frame_num%10==0){
            Mat b,g,r;
            //  imshow("frameWindow",frame);
            //  waitKey(0);
            calcHistogram(frame,b,g,r);
            f<<frame_num<<endl; 
            for(int i=0;i<256;i++)
                f<<b.at<float>(i,0)<<" ";
            f<<"\n";
            for(int i=0;i<256;i++)
                f<<g.at<float>(i,0)<<" ";
            f<<"\n";
            for(int i=0;i<256;i++)
                f<<r.at<float>(i,0)<<" ";
            f<<"\n";
        }
    }
    f.close(); 
    numEpiFrames = frame_num;
    cout<<"Frames in Episode:"<<frame_num<<endl;
}

void svcProcessing(char *svcName){
    bool svcStatus = LoadSVC(svcName);
    int svcSize = svcVector.size();
    fstream f;
    f.open("sampleSVC_1FP10F.txt",fstream::out);
    for(int i=0;i<svcSize;i++){
        Mat b,g,r;
        calcHistogram(svcVector[i],b,g,r);
        svcHistogram.push_back(b);
        svcHistogram.push_back(g);
        svcHistogram.push_back(r);
        f<<i<<endl; 
        for(int i=0;i<256;i++)
            f<<b.at<float>(i,0)<<" ";
        f<<"\n";
        for(int i=0;i<256;i++)
            f<<g.at<float>(i,0)<<" ";
        f<<"\n";
        for(int i=0;i<256;i++)
            f<<r.at<float>(i,0)<<" ";
        f<<"\n";
    }
    f.close(); 
}


void parseString(string s,Mat &v){
    stringstream ss;
    ss<<s;
    float x;
    for(int i=0;i<256;i++){
        ss>>x;
        v.at<float>(0,i) = x;
    }
}

void EpiSvcHistDist(){

    int method = CV_COMP_CORREL;
    memset(filter_I_selected,0,sizeof(filter_I_selected));
    int svcSize = svcVector.size();
    fstream f;
    f.open("sample_1FP10F.txt",fstream::in);
    int l_nos = 0;
    int curr_frame_nos;
    Mat b(256,1,CV_32FC1),g(256,1,CV_32FC1),r(256,1,CV_32FC1);
    while(f.good()){
        string line;
        getline(f,line);
        if(l_nos%4==0)
            curr_frame_nos = atoi(line.c_str());
        if(l_nos%4==1)
            parseString(line,b);
        if(l_nos%4==2)
            parseString(line,g);
        if(l_nos%4==3)
            parseString(line,r);
        l_nos++;
        if(l_nos%4==0){
            double ovScore = 0;
            for(int i=0;i<svcSize;i++){
                double sB = compareHist(svcHistogram[i*3+0],b,method);
                double sG = compareHist(svcHistogram[i*3+1],g,method);
                double sR = compareHist(svcHistogram[i*3+2],r,method);
                ovScore = sB*sG*sR > ovScore ? sB*sG*sR:ovScore;
            }
            histScore.push(make_pair(curr_frame_nos,ovScore));
        }
    }

    int thresh = histScore.size()/3;
    cout<<"Threshold"<<thresh<<" "<<histScore.size()<<endl;
    int selected = 0;
    while(!histScore.empty()){
        int f_select = histScore.top().first;
        if(selected<=thresh){
            if(filter_I_selected[f_select]==0){
                filter_I_selected[f_select]=1;
                selected++;
            }
        }
        histScore.pop();
    }
}

void write_filter_I(char *epiName){
    VideoCapture episode;
    episode.open(epiName);

    int frame_num=0;
    int selected = 0;
    Mat frame;
    fstream f;
    while(true){
        episode>>frame;
        if(frame.empty()){
            break;
        }
        frame_num++;
        string towrite = "courtneyMatch/"+to_string(frame_num)+".png";
        if(filter_I_selected[frame_num]==1){
            imshow("selected",frame);
            waitKey(5);
            selected++;
        }
    }
    f.close(); 
    cout<<"Selected:"<<selected<<endl;
}

int main(int argc, char **argv){
    if(argc != 3){
        cout<<"Check command line arguments\n";
        return -1;
    }

    VideoCapture episode;
    char *epiName = argv[1];
    char *svcName = argv[2];
    /* THIS IS FOR COLOR HISTOGRAM*/
    epiFilter_I_Processing(epiName);
    svcProcessing(svcName);
    EpiSvcHistDist();
    write_filter_I(epiName);
    return 0;
}
