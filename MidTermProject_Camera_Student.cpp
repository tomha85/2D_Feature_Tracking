
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
using namespace std;


int main(int argc, const char *argv[])
{
    string dataPath = "../";  
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; 
    string imgFileType = ".png";
    int imgStartIndex = 0; 
    int imgEndIndex = 9;   
    int imgFillWidth = 4;   
    int dataBufferSize = 2; 
    vector<DataFrame> dataBuffer; 
    bool bVis = false;

    ofstream dt_file,dt_matches,dt_time;
    dt_file.open ("../Keypoints.csv");    
    dt_matches.open ("../Matched_Keypoints.csv");   
    dt_time.open ("../Time.csv");    

    vector<string> detector_type_names = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
    vector<string> descriptor_type_names = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};
    
    for(auto detector_type_name:detector_type_names) 
    {
        bool write_file = false;

        for(auto descriptor_type_name:descriptor_type_names) 
        {
            if(detector_type_name.compare("AKAZE")==0 && descriptor_type_name.compare("AKAZE")==0)
                continue;
            if(detector_type_name.compare("AKAZE")!=0 && descriptor_type_name.compare("AKAZE")==0)
                continue;
            
            dataBuffer.clear();
            
            cout << "Detector Type: " << detector_type_name << " Descriptor Type: " << descriptor_type_name << endl;
            
            if(!write_file)
            {
                dt_file << detector_type_name;
            }                
            
            dt_matches << detector_type_name << "_" << descriptor_type_name;            
            dt_time << detector_type_name << "_" << descriptor_type_name;

            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
            {
                double t = (double)cv::getTickCount();
                
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;
                
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                DataFrame frame;
                frame.cameraImg = imgGray;

                if (dataBuffer.size() > dataBufferSize) 
                {
                    dataBuffer.erase(dataBuffer.begin());                    
                }                

                dataBuffer.push_back(frame);
                
                cout << "#1 : Load Images done" << endl;
                
                vector<cv::KeyPoint> keypoints;            
                string detectorType = detector_type_name;                 

                if (detectorType.compare("SHITOMASI") == 0)
                {
                    detKeypointsShiTomasi(keypoints, imgGray, false);
                }                
                else if (detectorType.compare("HARRIS") == 0) 
                {
                    detKeypointsHarris(keypoints, imgGray, false);
                }                
                else if (detectorType.compare("FAST")  == 0 ||detectorType.compare("BRISK") == 0 ||
                        detectorType.compare("ORB")   == 0 ||detectorType.compare("AKAZE") == 0 ||
                        detectorType.compare("SIFT")  == 0)
                {
                    detKeypointsModern(keypoints, imgGray, detectorType, false);
                }
                else
                {
                    throw invalid_argument(detectorType + " is not a valid detectorType. Try SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT.");
                }
               
                bool bFocusOnVehicle = true;
                cv::Rect vehicleRect(535, 180, 180, 150);
                vector<cv::KeyPoint>::iterator keypoint;
                vector<cv::KeyPoint> region;
                
                if (bFocusOnVehicle)
                {
                    for(keypoint = keypoints.begin(); keypoint != keypoints.end(); ++keypoint)
                    {
                        if (vehicleRect.contains(keypoint->pt))
                        {  
                            cv::KeyPoint newKeyPoint;
                            newKeyPoint.pt = cv::Point2f(keypoint->pt);
                            newKeyPoint.size = 1;
                            region.push_back(newKeyPoint);
                        }
                    }

                    keypoints =  region;
                    cout << "Region of image n= " << keypoints.size()<<" keypoints"<<endl;
                }
                
                if(!write_file)
                {
                    dt_file  << ", " << keypoints.size();
                }            
                          
                bool bLimitKpts = false;
                if (bLimitKpts)
                {
                    int maxKeypoints = 50;

                    if (detectorType.compare("SHITOMASI") == 0)
                    { 
                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                    }
                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                    cout << " NOTE: Keypoints have been limited!" << endl;
                }

                (dataBuffer.end() - 1)->keypoints = keypoints;
                cout << "#2 : Detect keypoints done" << endl;

                cv::Mat descriptors;             
                string descriptorType = descriptor_type_name; 
                descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
                
                (dataBuffer.end() - 1)->descriptors = descriptors;

                cout << "#3 : Extractor keypoints done" << endl;

                if (dataBuffer.size() > 1) 
                {
                    vector<cv::DMatch> matches;
                    string matcherType = "MAT_BF";    
                  
                    string descriptorType;
                    if (descriptorType.compare("SIFT") == 0) 
                    {
                        descriptorType == "DES_HOG";
                    }
                    else
                    {
                        descriptorType == "DES_BINARY";
                    }                    
                    
                    string selectorType = "SEL_KNN";    
                                       
                    matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                    (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                    matches, descriptorType, matcherType, selectorType);

                    
                    (dataBuffer.end() - 1)->kptMatches = matches;

                    dt_matches << ", " << matches.size();                   
                    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
                    dt_time << ", " << 1000*t;

                    cout << "#4 : Match keypoint descriptors done" << endl;

                    bVis = false; 
                    if (bVis)
                    {
                        cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                        cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                        (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                        matches, matchImg,
                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        string windowName = "Matching keypoints between two camera images";
                        cv::namedWindow(windowName, 7);
                        cv::imshow(windowName, matchImg);
                        cout << "Press key to continue to next image" << endl;
                        cv::waitKey(0); 
                    }
                    bVis = false;
                }

            } 
            if(!write_file)
            {
                dt_file << endl;   
            }            
            write_file = true;
            dt_matches << endl;
            dt_time << endl;
        }
    }
    dt_file.close();
    dt_matches.close();
    dt_time.close();   

    return 0;
}
//Looking the Matched_KeyPoints.csv and Time.csv file. There are 3 best choices
//for detect keypoint on vehicle
//First: FAST/BRIEF: 224-260 Keypoints in 14-16ms
//Second: FAST/ORB: 218-239 Keypoints in 14-16ms
//Third: FAST/SIFT: 232-259 Keypoints in 28-30ms
