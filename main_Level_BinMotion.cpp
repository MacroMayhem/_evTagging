/// @file main_Level_BinMotion.cpp
/// @Synopsis  
/// @author Aditya Singh <aditya.singh@research.iiit.ac.in>
/// @version 1.0
/// @date 2014-11-18
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

//uncomment to view images as they are processed 
//#define SHOW

struct comparePQ{
    bool operator()(pair<int,double>p1,pair<int,double>p2){
        return p1.second>p2.second;
    }   
};

priority_queue<pair<int,double>,vector<pair<int,double> >,comparePQ>BinMotionScore;
vector<uint64_t>svcDescriptor;
vector<Mat>svcVector;
bool selected_Frames_BinMotion[1000000];
int DIVISION ;
int scores2pick;
int frameSkipWindow;

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

uint64_t BMFeatureExtractor(Mat &m1,Mat &m2){
    //int toIgnore = 0;
    Mat m1_g,m2_g;
    cvtColor(m1,m1_g,CV_BGR2GRAY);
    cvtColor(m2,m2_g,CV_BGR2GRAY);

    Mat sub;
    //cout<<m1.size()<<" "<<m2.size()<<endl;
    //absdiff(m1_g,m2_g,sub);
    // imshow("m1",m1);
    // imshow("m2",m2);
    // waitKey(0);
    int rs=m1_g.rows,cs=m1_g.cols;
    int pixPerArea = (rs*cs)/(DIVISION*DIVISION);

    vector<int>diffCount(DIVISION*DIVISION,0);
    int dr=rs/DIVISION+1;
    int dc=cs/DIVISION+1; 
    for(int r=0;r<rs;r++){
        for(int c=0;c<cs;c++){
            int val = abs(int(m1_g.at<uchar>(r,c) - int(m2_g.at<uchar>(r,c))));
            int section = (r/dr)*(DIVISION)+c/dc;
            if(val)
                diffCount[section]++;
        }
    }
    uint64_t BM=0;
    for(int i=0;i<DIVISION*DIVISION;i++){
        if(diffCount[i]>=(pixPerArea/2)){
            BM|=1<<i;
        }
    }
    //    cout<<BM<<endl;
    //     imshow("sub",sub);
    //     waitKey(0);
    return BM;
}

void processEpi(string epiName){
    VideoCapture bm_video;
    bm_video.open(epiName);

    string fileName = name2write(epiName);
    fileName += "_BinMotion.txt";
    cout<<"EPI DESC Written to "<<fileName<<endl;
    fstream f;
    f.open(fileName,fstream::out);

    if(!bm_video.isOpened())
        return;
    Mat frame;
    int frame_num=0;
    Mat prevFrame;
    while(true){
        bm_video>>frame;
        frame_num++;
        if(frame.empty())
            break;
        if(frame_num%frameSkipWindow==0 && frame_num != 0){
            uint64_t val = BMFeatureExtractor(frame,prevFrame);
            f<<val<<endl;
            prevFrame = frame.clone();
        }
        if(frame_num==1)
            prevFrame = frame.clone();
    }
    cout<<"Frame in Video :"<<frame_num<<endl;
    f.close();
}

void processSVC(string svcName){
    VideoCapture svcVideo;
    svcVideo.open(svcName);
    string towrite = name2write(svcName);
    towrite+="_SVC_BinMotion.txt";
    cout<<"SVC DESC Written to "<<towrite<<endl;
    fstream f;
    f.open(towrite,fstream::out);

    int frame_num=0;
    Mat prevFrame;
    while(true){
        Mat frame;
        svcVideo>>frame;
        if(frame.empty())
            break;
        svcVector.push_back(frame);
        frame_num++;
        if(frame_num >= frameSkipWindow){
            uint64_t val = BMFeatureExtractor(frame,svcVector[frame_num-frameSkipWindow]);
            svcDescriptor.push_back(val);
            f<<frame_num<<" "<<val<<endl;
        }
    }
    f.close();
    cout<<"Frames in SVC:"<<frame_num<<endl;
    scores2pick = frame_num/frameSkipWindow;
    cout<<scores2pick<<":Scores 2 pick\n";
}

void getRowsCols(string s, int &r, int &c){
    stringstream ss;
    ss<<s;
    ss>>r;
    ss>>c;
}

bool compareHigh(int p1,int p2){
    if(p1>p2)
        return true;
    else return false;
}

void EpiSvcBMDistance(string epiName,string svcName){
    int svcDSize =svcDescriptor.size();
    fstream f,o;
    string matchOutFile = name2write(epiName)+name2write(svcName);
    matchOutFile += "_Score_BinMotion.txt";
    cout<<"Scores Written to "<<matchOutFile<<endl;
     
    string fileName = name2write(epiName);
    fileName ="./TEXTRESULTS/"+fileName+ "_BinMotion.txt";
    cout<<fileName<<endl;
    f.open(fileName,fstream::in);
    o.open(matchOutFile,fstream::out);
    int temp;
    int frame_nos = 0;
    int thresh = 200;
    queue<uint64_t>QscoreEPI; 
    int picked=0;

    while(f.good()){
        string line;
        getline(f,line);
        uint64_t val = strtoul(line.c_str(),NULL,0);
        QscoreEPI.push(val);
        picked++;
        if(picked==scores2pick){
            frame_nos += frameSkipWindow;
            int QepiSize = QscoreEPI.size();
            int endFrameNum = svcDSize-(QepiSize-1)*frameSkipWindow;
            int minBitDiff = 64*scores2pick;
            for(int i=0;i<scores2pick;i++){
                int currDiff=0;
                for(int j=0;j<scores2pick;j++){
                    uint64_t epiS = QscoreEPI.front();
                    QscoreEPI.pop();
                    QscoreEPI.push(epiS);
                    int svcIndex = i+j*frameSkipWindow;
                    uint64_t svcS = svcDescriptor[svcIndex];
                    uint64_t diffBits=svcS^epiS;
                    bitset<64>nos(diffBits);
                    currDiff+=nos.count();
                }
                if(currDiff<minBitDiff)
                    minBitDiff = currDiff;
            }
            BinMotionScore.push(make_pair(frame_nos-frameSkipWindow,minBitDiff));
            QscoreEPI.pop();
            picked--;
        }
    }
    cout<<"BMSIZE:"<<BinMotionScore.size()<<endl;
    int r;
    int thresh_selected = 0;
    while(!BinMotionScore.empty()){
        int sFrame = BinMotionScore.top().first;
        int sScore = BinMotionScore.top().second;
        if(selected_Frames_BinMotion[sFrame]==0 && thresh_selected <= thresh){
            selected_Frames_BinMotion[sFrame]=1;
            thresh_selected++;
        }
       o<<sFrame<<" "<<sScore<<endl;
       BinMotionScore.pop();
    }
        o.close();
}


    void viewSelection(string epiName){
        VideoCapture bm_video;
        bm_video.open(epiName);

        if(!bm_video.isOpened())
            return;
        Mat frame;
        int frame_num=0;
        while(true){
            bm_video>>frame;
            if(frame.empty())
                break;
            if(selected_Frames_BinMotion[frame_num]){
                imshow("selected",frame);
                waitKey(0);
            }
            frame_num++;
        }
    }


    int main(int argc, char*argv[]){
        if(argc != 4){
            cout<<"eg: ./exec episodeName svcName DIVISIONS\n";
            return 0;
        }
        string epiName(argv[1]);
        string svcName(argv[2]);

        frameSkipWindow = 5;   // frames to skip per VIDEO

        stringstream ss;
        ss<<argv[3];
        ss>>DIVISION;

        processEpi(epiName);
        //processSVC(svcName);
        //EpiSvcBMDistance(epiName,svcName);
        //viewSelection(epiName);
    }
