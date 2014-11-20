#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
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

#define printStatus 1

bool CompareDV(pair<int,float>p1,pair<int,float>p2){
 if(p1.second<p2.second)
   return true;
 else
   return false;
}

//void priority_queue<pair<int,float>,vector<pair<int,float> >,CompareDV>pQ;
queue<Mat>epiQ;
vector<Mat> gifVector;

void frameDetails(Mat &frame){
  int rows = frame.rows;
  int cols = frame.cols;
  cout<<cols<<" "<<rows<<endl;
}



bool LoadGif(char *gifName){
  VideoCapture gif;
  gif.open(gifName);
  if(!gif.isOpened())
   return 0;

  Mat frame;

  while(true){
    gif>>frame;
    if(frame.empty())
      break;
    gifVector.push_back(frame);
  }
  return 1;
}

int CompareSEQS(){
  queue<Mat>tempQ;
  int l = gifVector.size();
  while(!epiQ.empty()){
    Mat eFrame = epiQ.front();
    tempQ.push(eFrame);
    epiQ.pop();
  }
  while(!tempQ.empty()){
    Mat topush = tempQ.front();
    epiQ.push(topush);
    tempQ.pop();
  }
  epiQ.pop();
  return 1;

}


int main(int argc, char **argv){
  if(argc != 3){
    cout<<"Check command line arguments\n";
    return -1;
  }

  VideoCapture episode;
  char *epiName = argv[1];
  char *gifName = argv[2];
  episode.open(epiName);

  bool gifStatus = LoadGif(gifName);
  if(!episode.isOpened() && !gifStatus){
    cout<<"Couldn't load the Videos\n";
    return -1;
  }

  Mat frame;
  int frame_num=0;
  
  int gifSize = gifVector.size();
  
  if(printStatus)
  cout<<"frames in gif:"<<gifSize<<endl; 
  
  while(true){

    episode>>frame;
    if(frame.empty()){
      break;
    }
    if(!frame_num)
      frameDetails(frame);
    frame_num++;
    int epiSize = epiQ.size(); 
    // if(epiSize==gifSize){
  //   if(printStatus)
     // int val = CompareSEQS();
   //  }
     //epiQ.push(frame);
  }
  cout<<epiQ.size();
  if(printStatus)
  cout<<"Frames in Episode:"<<frame_num<<endl;
return 0;
}
