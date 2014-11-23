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
bool selected_Frames_BinMotion[10000000];
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

void edgeMap(Mat &img){
    Mat img2;
    GaussianBlur( img, img2, Size(3,3), 0, 0, BORDER_DEFAULT );
    Mat img_g;
    cvtColor(img2,img_g,CV_BGR2GRAY);
    
    int ddepth = CV_16S;
    int scale = 1;
    int delta = 0;
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat grad;
    Sobel(img_g, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    Sobel(img_g, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    threshold(grad, grad,50, 255,0); 
    int rs=img_g.rows,cs=img_g.cols;
    int pixPerArea = (rs*cs)/(DIVISION*DIVISION);

    int dr=rs/DIVISION;
    int dc=cs/DIVISION; 
    vector<Mat>edgeDivisionMap(DIVISION*DIVISION,Mat());
    for(int i=0;i<DIVISION;i++){
        for(int j=0;j<DIVISION;j++){
           edgeDivisionMap[i*DIVISION+j] = grad(Range(i*dr,min(i*dr+dr,rs-1)),Range(j*dc,min(j*dc+dc,cs-1))).clone();
        }
    }
    stringstream ss;
    string winName="EDGE";
    imshow("Main",grad);
   // for(int i=0;i<DIVISION*DIVISION;i++){
        //ss<<i;
        //winName+=ss.str();
        //imshow(winName.c_str(),edgeDivisionMap[i]);
   // }
        waitKey(0);
}
uint64_t BMFeatureExtractor(Mat &m1,Mat &m2){
    //int toIgnore = 0;
    Mat m1f,m2f;
    Mat m1_g,m2_g;
    //GaussianBlur(m1, m1f, Size( 5, 5), 0, 0 );
    //GaussianBlur(m2, m2f, Size( 5,5), 0, 0 );
    cvtColor(m1,m1_g,CV_BGR2GRAY);
    cvtColor(m2,m2_g,CV_BGR2GRAY);
    
    Mat sub;
    //cout<<m1.size()<<" "<<m2.size()<<endl;
    /* absdiff(m1_g,m2_g,sub);
      imshow("m1",m1f);
      imshow("m2",m2f);
      imshow("sub",sub);
      waitKey(0);*/
    int rs=m1_g.rows,cs=m1_g.cols;
    int pixPerArea = (rs*cs)/(DIVISION*DIVISION);
    vector<int>diffCount(DIVISION*DIVISION,0);
    int dr=rs/DIVISION+1;
    int dc=cs/DIVISION+1; 
    for(int r=0;r<rs;r++){
        for(int c=0;c<cs;c++){
            int val = abs(int(m1_g.at<uchar>(r,c) - int(m2_g.at<uchar>(r,c))));
            int section = (r/dr)*(DIVISION)+c/dc;
            if(val>50)
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

void testFunction(string epiName){
    VideoCapture epiVideo;
    epiVideo.open(epiName);
    int frame_num=0;
    Mat prevFrame;

    vector<Mat> epiVector;
    vector<int64_t>epiDescriptor;
    int epiStart = 91024;
    int epiLen = 100;
    while(true){
        Mat frame;
        Mat res_frame;
        epiVideo>>frame;
        if(frame.empty())
            break;
        if(frame_num>=epiStart && frame_num<=epiStart+epiLen &&frame_num%(frameSkipWindow-1)==0){
            cout<<frame_num<<endl;
               imshow("frame",frame);
               for(int i=0;i<svcVector.size();i++){
                    imshow("svc",svcVector[i]);
                    waitKey(20);
               }
        }

        if(frame_num>=epiStart-frameSkipWindow && frame_num<=epiStart+epiLen && frame_num%(frameSkipWindow-1)==0){
           resize(frame,res_frame,Size(frame.cols/2,frame.rows/2));
           epiVector.push_back(res_frame);
        }
        frame_num++;
    }
    for(int i=0;i<epiVector.size()-1;i++){
            uint64_t val = BMFeatureExtractor(epiVector[i+1],epiVector[i]);
            epiDescriptor.push_back(val);
    }
  // CHECK DISTMATCHING FUCNTION. TAKING 0-4 4-8 8-12 . looping needs to change . 
    int eSize = epiDescriptor.size();
    int sSize = svcDescriptor.size();
    for(int i=0;i<eSize-scores2pick;i++){
        cout<<"Starting at: "<<epiStart+i*(frameSkipWindow-1)<<endl;
        int minBitDiff = 64*scores2pick;
        for(int j=0;j<frameSkipWindow-1;j++){
            int currDiff=0;
            for(int k=0;k<scores2pick;k++){
                    uint64_t epiS = epiDescriptor[i+k];
                    int svcIndex = j+k*(frameSkipWindow-1);
                    uint64_t svcS = svcDescriptor[svcIndex];
                    uint64_t diffBits=svcS^epiS;
                    bitset<64>nos(diffBits);
                    currDiff+=nos.count();
                    cout<<epiS<<" "<<svcS<<" "<<nos.count()<<endl;
                    imshow("EPI1",epiVector[i+k]);
                    imshow("EPI2",epiVector[i+k+1]);
                    imshow("SVC1",svcVector[svcIndex]);
                    imshow("SVC2",svcVector[svcIndex+frameSkipWindow-1]);
                    waitKey(0);
            }
                cout<<currDiff<<" BEST "<<minBitDiff<<endl;
                if(currDiff<minBitDiff)
                    minBitDiff = currDiff;
        }
    }
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
    int prev_num=0;
    Mat prevFrame;
    int W,H;
    while(true){
        bm_video>>frame;
        if(frame.empty())
            break;
        if(frame_num!=0 && frame_num>=38483){
            Mat re_frame;
            resize(frame,re_frame,Size(W,H));
            uint64_t val = BMFeatureExtractor(re_frame,prevFrame);
            if(val!=0 && frame_num){
            edgeMap(prevFrame);
            prevFrame = re_frame.clone();
            f<<prev_num<<" "<<val<<endl;
            prev_num=frame_num;
            }
        }else{
            H=frame.rows/2;
            W=frame.cols/2;
            resize(frame,prevFrame,Size(W,H));
        }
        frame_num++;
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
    uint64_t val=0;
    int frame_num=0;
    Mat prevFrame;
    while(true){
        Mat frame;
        svcVideo>>frame;
        if(frame.empty())
            break;
        frame_num++;
        if(frame_num>=2)
           val = BMFeatureExtractor(frame,prevFrame);
        else{
            svcVector.push_back(frame);
            prevFrame = frame.clone();
        }
        if(val!=0){
            edgeMap(prevFrame);
            prevFrame = frame.clone();
            svcVector.push_back(frame);
            svcDescriptor.push_back(val);
            f<<frame_num<<" "<<val<<endl;
        }
    }
    f.close();
    cout<<"Frames in SVC:"<<frame_num<<endl;
//    scores2pick = (frame_num/(frameSkipWindow-1))-1;
    scores2pick = svcDescriptor.size();
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
    int thresh = 200;
    queue<uint64_t>QscoreEPI; 
    queue<int>Qdelta;
    int picked=0;
    while(f.good()){
        string line;
        getline(f,line);
        stringstream ss;
        int deltaNos;
        uint64_t val;//strtoul(line.c_str(),NULL,0);
        ss<<line;
        ss>>deltaNos;
        ss>>val;
        QscoreEPI.push(val);
        Qdelta.push(deltaNos);
        picked++;
        if(picked==scores2pick){
                int currDiff=0;
                for(int j=0;j<scores2pick;j++){
                    uint64_t epiS = QscoreEPI.front();
                    QscoreEPI.pop();
                    QscoreEPI.push(epiS);
                    uint64_t svcS = svcDescriptor[j];
                    uint64_t diffBits=svcS^epiS;
                    bitset<64>nos(diffBits);
                    currDiff+=nos.count();
                }
            int deltaFront = Qdelta.front();
            BinMotionScore.push(make_pair(deltaFront,currDiff));
            QscoreEPI.pop();
            Qdelta.pop();
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

/*0 index working. First frame of SVC is numbered 0. First frame of Epi is numbered 0. Epi are taken from 0-4, 4-8, 8-12
 * so on.svc Desc are computed for 0-4,1-5,2-6 so on. Now during comparision, Taking scores2pick nos of epiDesc svcDesc
 * are scored. EpiDes[0,1,2,...score2pick] vs for i=0 to 3:svcDesc(0+i,4+i,8+i,..,end)*/
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

             processSVC(svcName);
             processEpi(epiName);
         //   testFunction(epiName);
         //    EpiSvcBMDistance(epiName,svcName);
        //viewSelection(epiName);
    }
