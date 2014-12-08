/// @file FrameDump
/// @Synopsis  
/// @author Aditya Singh <aditya.singh@research.iiit.ac.in>
/// @version 1.0
/// @date 2014-12-8
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include "opencv2/core/core.hpp"
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
#include <cstring>
#include <queue>
#include <utility>
#include<cstdint>
#include<bitset>
using namespace std;
using namespace cv; 

int EH=360;
int EW=640;

void processEpi(string epiName){
    VideoCapture bm_video;
    bm_video.open(epiName);


    string fileName = name2write(epiName);
    string flowName = name2write(epiName);
    flowName+="_OPFLOW_HIST.txt";
    fileName += "_BinMotion.txt";
    cout<<"EPI DESC Writing to "<<fileName<<endl;
    cout<<"OP FLOW Writing to "<<flowName<<endl;
    fstream f,of;
    f.open(fileName,fstream::out);
    of.open(flowName,fstream::out);

    if(!bm_video.isOpened())
        return;
    Mat frame;
    int frame_num=0;
    int prev_num=0;
    Mat prevFrame;
    int W,H;
    Mat of_hist(1,9,CV_32FC1,Scalar(0));
    while(true){
        bm_video>>frame;
        if(frame.empty())
            break;
               if(frame_num!=0){
            Mat re_frame;
            resize(frame,re_frame,Size(eW,eH));
            uint64_t val = BMFeatureExtractor(prevFrame,re_frame);
            if(val!=0){
                prevFrame = re_frame.clone();
                f<<prev_num<<" "<<val<<endl;
                EPI_frames_key[prev_num]=1;
                prev_num=frame_num;
            }
        }
        else{
            resize(frame,prevFrame,Size(eW,eH));
        }
        frame_num++;
    }
    cout<<"Frame in Video :"<<frame_num<<endl;
    of.close();
    f.close();
}


int main(int argc, char*argv[]){
    if(argc != 4){
        cout<<"eg: ./exec InputList.txt OutputFolder\n";
        return 0;
    }
    string epiList(argv[1]);
    processEpi(epiName);
}
