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

priority_queue<pair<int,double>,vector<pair<int,double> >,comparePQ>PQ_LevelFinal;
vector<uint64_t>svcDescriptor;
vector<Mat>svcOpticalDescriptor;
vector<Mat>svcVector;
bool selected_Frames[10000000];
int DIVISION ;
int scores2pick;



//** OPTICAL FLOW HISTOGRAM SETTING
double pyr_scale = 0.5;
int levels = 2;
int winsize = 10;
int iterations = 2;
int poly_n = 7;
double poly_sigma = 1.5;
int opFlowFlag = OPTFLOW_FARNEBACK_GAUSSIAN;
float PI = 3.14159265;
//** NOT REQUIRED
int frameSkipWindow;



//** FOR EDGE RELATED TESTING
vector<Mat>edgeTestSVC,edgeTestEPI;


//** FOR VISUALISATION STUFF TESTING
int counter = 0;
string init="S";

int eH=360;
int eW=640;
int sH=180;
int sW=320;

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

void testEdgeCriteria(){
    int s=edgeTestSVC.size(),e=edgeTestEPI.size();

    Size sz = Size(128,128);
    string E="E",S="S";
    stringstream ss;
    for(int i=0;i<s;i++){
        ss.str("");
        ss<<i;
        string path = S+ss.str()+".png";
        resize(edgeTestSVC[i],edgeTestSVC[i],sz,0,0,INTER_LINEAR);
        imwrite(path.c_str(),edgeTestSVC[i]);
        //imshow("svc",edgeTestSVC[i]);
        // waitKey(0);
    }
    for(int i=0;i<e;i++){
        ss.str("");
        ss<<i;
        string path = E+ss.str()+".png";
        resize(edgeTestEPI[i],edgeTestEPI[i],sz,0,0,INTER_LINEAR);
        imwrite(path.c_str(),edgeTestEPI[i]);
        // imshow("epi",edgeTestEPI[i]);
        // waitKey(0);
    }
}


void opticalFlowHistogram(Mat img_prev, Mat img_next,vector<float>&OpticalFlowHistogram){
    int binDelta = 225;
    double thresh = 1.5;
    double Nos=0;
    Mat img_prev_g,img_next_g;
    cvtColor(img_prev,img_prev_g,CV_BGR2GRAY);
    cvtColor(img_next,img_next_g,CV_BGR2GRAY);
    Mat oFlow,xyFlow[2];
    calcOpticalFlowFarneback(img_prev_g,img_next_g,oFlow,pyr_scale,levels,winsize,iterations,poly_n,poly_sigma,opFlowFlag);
    int rs = oFlow.rows;
    int cs = oFlow.cols;
    split(oFlow,xyFlow);
    Mat newDummy=img_next.clone();
    for(int r=0;r<rs;r+=10){
        for(int c=0;c<cs;c+=10){
            Point p1(c,r);
            Point p2;
            double deltaX=xyFlow[0].at<float>(r,c),deltaY=xyFlow[1].at<float>(r,c);
            p2.x = c+deltaX;
            p2.y = r+deltaY;
            if(p2.x>=0 && p2.x<cs && p2.y>=0 && p2.y <= rs){
                Nos++;
            if(fabs(deltaX)<=thresh && fabs(deltaY)<=thresh){
                OpticalFlowHistogram[8]++;
                continue;
            }
            double theta = atan2(-1.0*deltaY,deltaX)*180.0/PI;
            if(theta<0)
                theta+=360.0;
            int roundTheta = theta*10;
            int binVal = roundTheta/binDelta;
            if(binVal%2 != 0){
                      if(roundTheta%binDelta==0){
                      OpticalFlowHistogram[(binVal/2+1)%8]+=0.5;
                      OpticalFlowHistogram[(binVal/2-1+8)%8]+=0.5;
                      }else{
                      OpticalFlowHistogram[(binVal/2+1)%8]+=1;
                      }
            }else
                      OpticalFlowHistogram[(binVal/2)%8]+=1;

           // line(newDummy,p1,p2,Scalar(255,0,0),1,4,0);
           // circle(newDummy,p2,2,Scalar(255,0,0),1,8,0);
            //circle(newDummy,p2,2,Scalar(255,0,0),1,8,0);
            }
        }
    }
    Nos = Nos > 1?Nos:1;
    for(int i=0;i<9;i++){
        OpticalFlowHistogram[i]/=Nos;
       // cout<<i%8<<"->"<<OpticalFlowHistogram[i]<<"   ";
    }
      //cout<<endl;
      //imshow("prev",img_prev);
      //imshow("next",newDummy);
      //waitKey(0);
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
    if(init=="S")
        edgeTestSVC.push_back(grad);
    else
        edgeTestEPI.push_back(grad);

    // imshow("Main",grad);
    // for(int i=0;i<DIVISION*DIVISION;i++){
    //ss<<i;
    //winName+=ss.str();
    //imshow(winName.c_str(),edgeDivisionMap[i]);
    // }
    // waitKey(0);
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
    /*  absdiff(m1_g,m2_g,sub);
        imshow("m1",m1);
        imshow("m2",m2);
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
            if(val>10)
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
    init+="E";
    vector<float>OFHistogram(9,0.0);
    while(true){
        bm_video>>frame;
        if(frame.empty())
            break;
        /*if(frame_num >= 38596 && frame_num<=38625){
          Mat re_frame;
          resize(frame,re_frame,Size(sW,sH));
          uint64_t val = BMFeatureExtractor(re_frame,prevFrame);
          prevFrame = re_frame.clone();
        //edgeMap(re_frame);

        }*/
        if(frame_num!=0){
            Mat re_frame;
            resize(frame,re_frame,Size(eW,eH));
            uint64_t val = BMFeatureExtractor(prevFrame,re_frame);
            if(val!=0){
                opticalFlowHistogram(prevFrame,re_frame,OFHistogram);
                for(int i=0;i<9;i++){
                    of<<OFHistogram[i]<<" ";
                    OFHistogram[i]=0;
                }
                of<<endl;
                prevFrame = re_frame.clone();
                f<<prev_num<<" "<<val<<endl;
                prev_num=frame_num;
            }
        }
        else{
            //H=frame.rows/2;
            //W=frame.cols/2;
            resize(frame,prevFrame,Size(eW,eH));
            //  prevFrame = frame.clone();
        }
        frame_num++;
    }
    cout<<"Frame in Video :"<<frame_num<<endl;
    of.close();
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

    for(int i=0;i<9;i++){
    }

    while(true){
        Mat frame;
        svcVideo>>frame;
        if(frame.empty())
            break;
        frame_num++;
        if(frame_num>=2){
            resize(frame,frame,Size(sW,sH));
            val = BMFeatureExtractor(prevFrame,frame);
        }
        else{
            resize(frame,frame,Size(sW,sH));
            svcVector.push_back(frame.clone());
            prevFrame = frame.clone();
        }
        if(val!=0){
            //edgeMap(prevFrame);
            vector<float>OFHistogram(9,0.0);
            opticalFlowHistogram(prevFrame,frame,OFHistogram);
            Mat OFlow(1,9,CV_32FC1,Scalar(0.0));
            for(int i=0;i<9;i++){
                OFlow.at<float>(0,i)=OFHistogram[i];
            }
            svcOpticalDescriptor.push_back(OFlow);
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

void FinalEVAL(string epiName, string svcName, string addON){
    string matchOutFile = name2write(epiName)+name2write(svcName)+addON+".txt";
    fstream o;
    o.open(matchOutFile,fstream::out);
    int thresh = 500;
    float lastVal=-1;
    int thresh_selected = 0;

    while(!PQ_LevelFinal.empty()){
        int sFrame = PQ_LevelFinal.top().first;
        float sScore = PQ_LevelFinal.top().second;
        //if(selected_Frames_BinMotion[sFrame]==0 && thresh_selected <= thresh){
         //   selected_Frames_BinMotion[sFrame]=1;
          //  thresh_selected++;
       // }
        o<<sFrame<<" "<<sScore<<endl;
        PQ_LevelFinal.pop();
    }
    o.close();
}

void Level_BM(string epiName,string svcName){
    int svcDSize =svcDescriptor.size();
    fstream f;

    string fileName = name2write(epiName);
    fileName ="./TEXTRESULTS/"+fileName+ "_BinMotion.txt";
    cout<<fileName<<" "<<endl;
    f.open(fileName,fstream::in);
    int temp;
    int thresh = 200;
    queue<uint64_t>QscoreEPI; 
    queue<int>Qdelta;
    int picked=0;
    while(f.good()){
        string line,ofLine;
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
            float currDiff=0;
            double currOFscore=0;
            for(int j=0;j<scores2pick;j++){
                //** BINARY MOTION 
                uint64_t epiS = QscoreEPI.front();
                QscoreEPI.pop();
                QscoreEPI.push(epiS);
                uint64_t svcS = svcDescriptor[j];
                uint64_t diffBits=svcS^epiS;
                bitset<64>nos(diffBits);
                currDiff+=nos.count();
            }
            float normaliser = DIVISION*DIVISION*picked;
            float overAllScore = currDiff/normaliser;
            int deltaFront = Qdelta.front();
            PQ_LevelFinal.push(make_pair(deltaFront,overAllScore));
            QscoreEPI.pop();
            Qdelta.pop();
            picked--;
        }
    }
    f.close();
    cout<<"BMSIZE:"<<PQ_LevelFinal.size()<<endl;
    FinalEVAL(epiName,svcName,"_Score_BM");
}

void Level_OF(string epiName,string svcName){
    int svcDSize =svcDescriptor.size();
    fstream f,of;

    string fileName = name2write(epiName);
    fileName ="./TEXTRESULTS/"+fileName+ "_BinMotion.txt";
    string flowName =name2write(epiName);
    flowName="./TEXTRESULTS/"+flowName+"_OPFLOW_HIST.txt";
    cout<<flowName<<endl;
    f.open(fileName,fstream::in);
    of.open(flowName,fstream::in);
    int temp;
    int thresh = 200;
    queue<Mat>QoFlow;
    queue<uint64_t>QscoreEPI; 
    queue<int>Qdelta;
    int picked=0;
    while(f.good()){
        string line,ofLine;
        getline(f,line);
        getline(of,ofLine);
        stringstream ss, ofss;
        int deltaNos;
        uint64_t val;//strtoul(line.c_str(),NULL,0);
        ss<<line;
        ss>>deltaNos;
        ss>>val;
        ofss<<ofLine;
        Mat ofMat(1,9,CV_32FC1,Scalar(0.0));
        for(int i=0;i<9;i++){
            float nos;
            ofss>>nos;
            ofMat.at<float>(0,i)=nos;
        }
        QoFlow.push(ofMat);
        Qdelta.push(deltaNos);
        picked++;
        if(picked==scores2pick){
            double currOFscore=0;
            for(int j=0;j<scores2pick;j++){
                Mat ofCurrent = QoFlow.front();
                QoFlow.pop();
                QoFlow.push(ofCurrent);
                currOFscore+= compareHist(ofCurrent,svcOpticalDescriptor[j],CV_COMP_CORREL);
            }
            float overAllScore = 1.0-currOFscore/float(picked);
            int deltaFront = Qdelta.front();
            PQ_LevelFinal.push(make_pair(deltaFront,overAllScore));
            Qdelta.pop();
            QoFlow.pop();
            picked--;
        }
    }
    f.close();
    of.close();
    cout<<"BMSIZE:"<<PQ_LevelFinal.size()<<endl;
    FinalEVAL(epiName,svcName,"_Score_OF");
}

void Level_BM_OF(string epiName,string svcName){
    int svcDSize =svcDescriptor.size();
    fstream f,of;

    string fileName = name2write(epiName);
    fileName ="./TEXTRESULTS/"+fileName+ "_BinMotion.txt";
    string flowName =name2write(epiName);
    flowName="./TEXTRESULTS/"+flowName+"_OPFLOW_HIST.txt";
    cout<<fileName<<" "<<flowName<<endl;
    f.open(fileName,fstream::in);
    of.open(flowName,fstream::in);
    int temp;

    vector<pair<int, double> >thresh_BMs;
    priority_queue<pair<int,double>,vector<pair<int,double> >,comparePQ>PQ_BMs;
    map<int,int> index_BMs;
    int num_BMs=0;

    queue<Mat>QoFlow;
    queue<uint64_t>QscoreEPI; 
    queue<int>Qdelta;
    int picked=0;
    while(f.good()){
        string line,ofLine;
        getline(f,line);
        getline(of,ofLine);
        stringstream ss, ofss;
        int deltaNos;
        uint64_t val;//strtoul(line.c_str(),NULL,0);
        ss<<line;
        ss>>deltaNos;
        ss>>val;
        ofss<<ofLine;
        Mat ofMat(1,9,CV_32FC1,Scalar(0.0));
        for(int i=0;i<9;i++){
            float nos;
            ofss>>nos;
            ofMat.at<float>(0,i)=nos;
        }
        QoFlow.push(ofMat);
        QscoreEPI.push(val);
        Qdelta.push(deltaNos);
        picked++;
        if(picked==scores2pick){
            float BM_Score=0;
            double OF_Score=0;
            for(int j=0;j<scores2pick;j++){
                //** BINARY MOTION 
                uint64_t epiS = QscoreEPI.front();
                QscoreEPI.pop();
                QscoreEPI.push(epiS);
                uint64_t svcS = svcDescriptor[j];
                uint64_t diffBits=svcS^epiS;
                bitset<64>nos(diffBits);
                BM_Score+=nos.count();

                //** OPTICAL FLOW HISTOGRAM
                Mat ofCurrent = QoFlow.front();
                QoFlow.pop();
                QoFlow.push(ofCurrent);
                OF_Score+= compareHist(ofCurrent,svcOpticalDescriptor[j],CV_COMP_CORREL);
            }
            float normaliser = DIVISION*DIVISION*picked;
            OF_Score = 1.0-OF_Score/float(picked);
            int deltaFront = Qdelta.front();
            PQ_BMs.push(make_pair(deltaFront,BM_Score/normaliser));
            thresh_BMs.push_back(make_pair(deltaFront,OF_Score));
            index_BMs[deltaFront]=num_BMs;
            num_BMs++;
            QscoreEPI.pop();
            Qdelta.pop();
            QoFlow.pop();
            picked--;
        }
    }
    cout<<"OFSIZE:"<<PQ_BMs.size()<<endl;
    f.close();
    of.close();
  
    int thresh_selected=0;
    int thresh = 1000;
    float last_score_val=-1;
    while(!PQ_BMs.empty()){
        int FID = PQ_BMs.top().first;
        float sScore = PQ_BMs.top().second;
        if(thresh_selected <= thresh){
            PQ_LevelFinal.push(thresh_BMs[index_BMs[FID]]);
            thresh_selected++;
            last_score_val = sScore; 
        }else{
            if(last_score_val == sScore){
            PQ_LevelFinal.push(thresh_BMs[index_BMs[FID]]);
            }
        }
        PQ_BMs.pop();
    }
    cout<<"BMSIZE:"<<PQ_LevelFinal.size()<<endl;
    FinalEVAL(epiName,svcName,"_Score_BM_OF");
}

void Level_OF_BM(string epiName,string svcName){
    int svcDSize =svcDescriptor.size();
    fstream f,of;

    string fileName = name2write(epiName);
    fileName ="./TEXTRESULTS/"+fileName+ "_BinMotion.txt";
    string flowName =name2write(epiName);
    flowName="./TEXTRESULTS/"+flowName+"_OPFLOW_HIST.txt";
    cout<<fileName<<" "<<flowName<<endl;
    f.open(fileName,fstream::in);
    of.open(flowName,fstream::in);
    int temp;

    vector<pair<int, double> >thresh_OFs;
    priority_queue<pair<int,double>,vector<pair<int,double> >,comparePQ>PQ_OFs;
    map<int,int> index_OFs;
    int num_OFs=0;

    queue<Mat>QoFlow;
    queue<uint64_t>QscoreEPI; 
    queue<int>Qdelta;
    int picked=0;
    while(f.good()){
        string line,ofLine;
        getline(f,line);
        getline(of,ofLine);
        stringstream ss, ofss;
        int deltaNos;
        uint64_t val;//strtoul(line.c_str(),NULL,0);
        ss<<line;
        ss>>deltaNos;
        ss>>val;
        ofss<<ofLine;
        Mat ofMat(1,9,CV_32FC1,Scalar(0.0));
        for(int i=0;i<9;i++){
            float nos;
            ofss>>nos;
            ofMat.at<float>(0,i)=nos;
        }
        QoFlow.push(ofMat);
        QscoreEPI.push(val);
        Qdelta.push(deltaNos);
        picked++;
        if(picked==scores2pick){
            float BM_Score=0;
            double OF_Score=0;
            for(int j=0;j<scores2pick;j++){
                //** BINARY MOTION 
                uint64_t epiS = QscoreEPI.front();
                QscoreEPI.pop();
                QscoreEPI.push(epiS);
                uint64_t svcS = svcDescriptor[j];
                uint64_t diffBits=svcS^epiS;
                bitset<64>nos(diffBits);
                BM_Score+=nos.count();

                //** OPTICAL FLOW HISTOGRAM
                Mat ofCurrent = QoFlow.front();
                QoFlow.pop();
                QoFlow.push(ofCurrent);
                OF_Score+= compareHist(ofCurrent,svcOpticalDescriptor[j],CV_COMP_CORREL);
            }
            float normaliser = DIVISION*DIVISION*picked;
            OF_Score = 1.0-OF_Score/float(picked);
            int deltaFront = Qdelta.front();
            thresh_OFs.push_back(make_pair(deltaFront,BM_Score/normaliser));
            PQ_OFs.push(make_pair(deltaFront,OF_Score));
            index_OFs[deltaFront]=num_OFs;
            num_OFs++;
            QscoreEPI.pop();
            Qdelta.pop();
            QoFlow.pop();
            picked--;
        }
    }
    cout<<"OFSIZE:"<<PQ_OFs.size()<<endl;
    f.close();
    of.close();
  
    int thresh_selected=0;
    int thresh = 1000;
    float last_score_val=-1;
    while(!PQ_OFs.empty()){
        int FID = PQ_OFs.top().first;
        float sScore = PQ_OFs.top().second;
        if(thresh_selected <= thresh){
            PQ_LevelFinal.push(thresh_OFs[index_OFs[FID]]);
            thresh_selected++;
            last_score_val = sScore; 
        }else{
            if(last_score_val == sScore){
            PQ_LevelFinal.push(thresh_OFs[index_OFs[FID]]);
            }
        }
        PQ_OFs.pop();
    }
    cout<<"BMSIZE:"<<PQ_LevelFinal.size()<<endl;
    FinalEVAL(epiName,svcName,"_Score_OF_BM");
}

void Level_BMOF(string epiName,string svcName){
    int svcDSize =svcDescriptor.size();
    fstream f,of;

    string fileName = name2write(epiName);
    fileName ="./TEXTRESULTS/"+fileName+ "_BinMotion.txt";
    string flowName =name2write(epiName);
    flowName="./TEXTRESULTS/"+flowName+"_OPFLOW_HIST.txt";
    cout<<fileName<<" "<<flowName<<endl;
    f.open(fileName,fstream::in);
    of.open(flowName,fstream::in);
    int temp;
    int thresh = 200;
    queue<Mat>QoFlow;
    queue<uint64_t>QscoreEPI; 
    queue<int>Qdelta;
    int picked=0;
    while(f.good()){
        string line,ofLine;
        getline(f,line);
        getline(of,ofLine);
        stringstream ss, ofss;
        int deltaNos;
        uint64_t val;//strtoul(line.c_str(),NULL,0);
        ss<<line;
        ss>>deltaNos;
        ss>>val;
        ofss<<ofLine;
        Mat ofMat(1,9,CV_32FC1,Scalar(0.0));
        for(int i=0;i<9;i++){
            float nos;
            ofss>>nos;
            ofMat.at<float>(0,i)=nos;
        }
        QoFlow.push(ofMat);
        QscoreEPI.push(val);
        Qdelta.push(deltaNos);
        picked++;
        if(picked==scores2pick){
            float currDiff=0;
            double currOFscore=0;
            for(int j=0;j<scores2pick;j++){
                //** BINARY MOTION 
                uint64_t epiS = QscoreEPI.front();
                QscoreEPI.pop();
                QscoreEPI.push(epiS);
                uint64_t svcS = svcDescriptor[j];
                uint64_t diffBits=svcS^epiS;
                bitset<64>nos(diffBits);
                currDiff+=nos.count();

                //** OPTICAL FLOW HISTOGRAM
                Mat ofCurrent = QoFlow.front();
                QoFlow.pop();
                QoFlow.push(ofCurrent);
                currOFscore+= compareHist(ofCurrent,svcOpticalDescriptor[j],CV_COMP_CORREL);
            }
            float normaliser = DIVISION*DIVISION*picked;
            float overAllScore = currDiff/normaliser+1.0-currOFscore/float(picked);
            int deltaFront = Qdelta.front();
            PQ_LevelFinal.push(make_pair(deltaFront,overAllScore));
            QscoreEPI.pop();
            Qdelta.pop();
            QoFlow.pop();
            picked--;
        }
    }
    cout<<"BMSIZE:"<<PQ_LevelFinal.size()<<endl;
    f.close();
    of.close();
    FinalEVAL(epiName,svcName,"_Score_BMOF");
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
        if(selected_Frames[frame_num]){
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
     // processEpi(epiName);
        processSVC(svcName);
     //   Level_BM(epiName,svcName);
     //     Level_OF(epiName,svcName);
    //    testEdgeCriteria();
    //   testFunction(epiName);
     //  Level_OF_BM(epiName,svcName);
       Level_BM_OF(epiName,svcName);
    //viewSelection(epiName);
}
