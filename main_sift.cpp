/// @file main_sift.cpp
/// @Synopsis  
/// @author Aditya Singh <aditya.singh@research.iiit.ac.in>
/// @version 1.0
/// @date 2014-11-18
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <ctype.h>
#include <unistd.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <queue>
#include <utility>
using namespace std;
using namespace cv; 

//uncomment to view images as they are processed 
//#define SHOW

struct comparePQ{
    bool operator()(pair<int,double>p1,pair<int,double>p2){
        return p1.second<p2.second;
    }   
};

priority_queue<pair<int,double>,vector<pair<int,double> >,comparePQ>featScore;
vector<Mat>svcDescriptor;
vector<Mat>svcVector;
bool selected_Frames[1000000];

void computeSIFT(Mat &src_img,Mat &descriptors,int nos_features){
    Mat image;
    cvtColor(src_img,image,CV_BGR2GRAY);

    vector<KeyPoint> keypoints;
    SiftFeatureDetector featureDetector(nos_features);
    SiftDescriptorExtractor featureExtractor(nos_features);    

    featureDetector.detect(image, keypoints);
    featureExtractor.compute(image,keypoints,descriptors);
#ifdef SHOW
    cout<<keypoints.size()<<" "<<descriptors.size()<<endl;
    Mat keypointImage;
    drawKeypoints(image, keypoints, keypointImage, Scalar::all(-1),DrawMatchesFlags::DEFAULT);
    imshow("Keypoints Found", keypointImage);
    waitKey(0);
#endif
}

string name2write(string filePath){
    int l = filePath.size();
    cout<<"assuming path to be *.mpg or something\n";
    string toret = "";
    bool dotFlag = 0;
    for(int i=l-1;i>=0;i--){
        if(filePath[i]=='/')
            break;
        if(dotFlag){
            toret+=filePath[i];
        }
        if(filePath[i]=='.')
            dotFlag = true;
    }
    l = toret.size();
    string rev_toret="";
    for(int i=l-1;i>=0;i--){
        rev_toret+=toret[i];
    }
    return rev_toret;
}

void processEpi(string epiName,string fdType){
    VideoCapture sift_video;
    initModule_nonfree(); 
    sift_video.open(epiName);

    string fileName = name2write(epiName);
    fileName += "_"+fdType +".txt";
    cout<<fileName<<endl;
    fstream f;
    f.open(fileName,fstream::out);

    if(!sift_video.isOpened())
        return;
    Mat frame;
    int frame_num=0;

    while(true){
        sift_video>>frame;
        if(frame.empty())
            break;
        frame_num++;
        if(frame_num%10==0){
            Mat descriptor;
            computeSIFT(frame,descriptor,100);
            int rd=descriptor.rows,cd= descriptor.cols;
            f<<rd<<" "<<cd<<endl;
            for(int r=0;r<rd;r++){
                for(int c=0;c<cd;c++){
                    f<<descriptor.at<float>(r,c)<<" ";
                }
                f<<endl;
            }
        }
    }

    f.close();
}

void processSVC(string svcName,string fdType){
    VideoCapture svcVideo;
    svcVideo.open(svcName);

    while(true){
        Mat frame;
        svcVideo>>frame;
        Mat descriptor;
        if(frame.empty())
            break;
        svcVector.push_back(frame);
        computeSIFT(frame,descriptor,100);
        svcDescriptor.push_back(descriptor);
    }
}

void getRowsCols(string s, int &r, int &c){
    stringstream ss;
    ss<<s;
    ss>>r;
    ss>>c;
}

void viewSIFTmatch(Mat img_1, Mat img_2){

    Mat desc1,desc2;
    Mat image1,image2;
    cvtColor(img_1,image1,CV_BGR2GRAY);
    cvtColor(img_2,image2,CV_BGR2GRAY);

    vector<KeyPoint> keypoints1,keypoints2;
    SiftFeatureDetector featureDetector(100);
    SiftDescriptorExtractor featureExtractor(100);    

    featureDetector.detect(image1, keypoints1);
    featureExtractor.compute(image1,keypoints1,desc1);
    featureDetector.detect(image2, keypoints2);
    featureExtractor.compute(image2,keypoints2,desc2);

    FlannBasedMatcher matcher;
    vector< DMatch > matches;
    matcher.match(desc1, desc2, matches);

    double max_dist = 0; double min_dist = 100;
    for( int j = 0; j < desc1.rows; j++ )
    { double dist = matches[j].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    
    std::vector< DMatch > good_matches;
    for( int i = 0; i < desc1.rows; i++ ){ 
            good_matches.push_back( matches[i]); 
    }
    Mat img_matches;
    cout<<good_matches.size()<<"$";
    fflush(stdout);
    drawMatches( img_1, keypoints1, img_2, keypoints2,good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
          vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    imshow( "Good Matches", img_matches );
    waitKey(0);

}

int SIFTmatch(Mat &desc1, Mat &desc2){
    
    FlannBasedMatcher matcher;
    vector< DMatch > matches;
    matcher.match(desc1, desc2, matches);

    double max_dist = 0; double min_dist = 100;
    for( int j = 0; j < desc1.rows; j++ )
    { double dist = matches[j].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    int toret = 0;
    for( int i = 0; i < desc1.rows; i++ ){ 
        if( matches[i].distance <= max(2*min_dist, 0.02) ){
            toret++;
        }
    }
    return toret;
}

void EpiSvcSiftDist(string epiName,string fdType){
    memset(selected_Frames,0,sizeof(selected_Frames));
    int svcSize = svcDescriptor.size();
    fstream f;


    string fileName = name2write(epiName);
    fileName+="_"+fdType+".txt";
    f.open(fileName.c_str(),fstream::in);

    int l_nos = 0;
    int curr_frame_nos=0;
    bool descRead = false;
    float tmp_fl;
    Mat b(100,1,CV_32FC1);
    while(f.good()){
        string line;
        getline(f,line);
        curr_frame_nos += 10;
        int r,c;
        getRowsCols(line,r,c);
        if(r==0 || c==0)
            continue;
        Mat desc(r,c,CV_32FC1);
        stringstream ss;
        for(int i=0;i<r;i++){
            getline(f,line);
            ss<<line;
            for(int j=0;j<c;j++){
                ss>>tmp_fl;
                desc.at<float>(i,j)=tmp_fl;
            }
        }
        cout<<"$"<<curr_frame_nos<<" ";
        fflush(stdout);
        int ovScore = 0;
        for(int i=0;i<svcSize;i++){
            int clScore=SIFTmatch(desc,svcDescriptor[i]);
            if(clScore>ovScore)
                ovScore = clScore;
        }
        featScore.push(make_pair(curr_frame_nos,ovScore));
    }   
    int thresh = 100;
    cout<<"Threshold"<<thresh<<endl;
    int selected = 0;
    while(!featScore.empty()){
        int f_select = featScore.top().first;
        if(selected<=thresh){
            if(selected_Frames[f_select]==0){
                cout<<f_select<<" "<<featScore.top().second<<endl;
                selected_Frames[f_select]=1;
                selected++;
            }
        }
        featScore.pop();
    }

}
void viewSelection(string epiName, string fdType){
    VideoCapture epiVideo;
    epiVideo.open(epiName);

    int svcSize = svcVector.size();

    Mat frame;
    int frame_num=0;
    while(true){
        epiVideo>>frame;
        frame_num++;
        Mat descriptor;
        if(frame.empty())
            break;
        if(frame_num==45800){
            for(int i=0;i<svcSize;i++){
             viewSIFTmatch(frame,svcVector[i]);
            }
        }
    }
}

int main(int argc, char** argv){
    if(argc != 4){
        cout<<"eg: ./main episodeName svcName SIFT\n";
        return 0;
    }
    string epiName=argv[1];
    string svcName=argv[2];
    string feature2test = argv[3];

    //      processEpi(epiName,feature2test);
          processSVC(svcName,feature2test);
    //    EpiSvcSiftDist(epiName,feature2test);
    viewSelection(epiName,feature2test);
}
