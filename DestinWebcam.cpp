#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"

#include <alproxies/altexttospeechproxy.h>
#include <alproxies/alvideodeviceproxy.h>
#include <alproxies/almotionproxy.h>
#include <alproxies/alrobotpostureproxy.h>
#include <alvision/alimage.h>
#include <alvision/alvisiondefinitions.h>
#include <alerror/alerror.h>

//// DeSTIN Library
#include "DestinNetworkAlt.h"
#include "Transporter.h"
#include "stdio.h"
#include "unit_test.h"
#include "BeliefExporter.h"
#include "VideoSource.h"
#include <time.h>

// standard library
#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <fstream>

#include <math.h>
#include <algorithm>
#include <iterator>


using namespace AL;
using namespace std;
using namespace cv;

const std::string robotIp="192.168.0.107";

//*************************Destin Functions******************************************


void testNan(float * array, int len){
    for(int i = 0 ; i < len ; i++){
        if(isnan(array[i])){
            printf("input had nan\n");
            exit(1);
        }
    }
}

void convert(cv::Mat & in, float * out) {

    if(in.channels()!=1){
        throw runtime_error("Excepted a grayscale image with one channel.");
    }

    if(in.depth()!=CV_8U){
        throw runtime_error("Expected image to have bit depth of 8bits unsigned integers ( CV_8U )");
    }

    cv::Point p(0, 0);
    int i = 0 ;
    for (p.y = 0; p.y < in.rows; p.y++) {
        for (p.x = 0; p.x < in.cols; p.x++) {
            //i = frame.at<uchar>(p);
            //use something like frame.at<Vec3b>(p)[channel] in case of trying to support color images.
            //There would be 3 channels for a color image (one for each of r, g, b)
            out[i] = (float)in.at<uchar>(p) / 255.0;
            i++;
        }
    }
}

float * callImage(Mat &image)
{
    float * float_image=new float[256*256];
    convert(image, float_image);

    testNan(float_image, 256*256);

    return float_image;
}

void printFPS(bool print){
    // start = initial frame time
    // end = final frame time
    // sec = time count in seconds
    // set all to 0
    static double end, sec, start = 0;

    // set final time count to current tick count
    end = (double)cv::getTickCount();

    //
    if(start != 0){
        sec = (end - start) / getTickFrequency();
        if(print==true){
            printf("fps: %f\n", 1 / sec);
        }
    }
    start = (double)cv::getTickCount();
}


void process(const std::string & robotIp)
{
    /** Create a proxy to ALVideoDevice on the robot.*/
    ALVideoDeviceProxy camProxy(robotIp, 9559);
    ALRobotPostureProxy posture(robotIp, 9559);

    // sit down as initial postion
    posture.goToPosture("Crouch", 0.5f);

    /** Subscribe a client image requiring 320*240 and BGR colorspace.*/
    //const std::string clientName = camProxy.subscribe("test", kQVGA, kBGRColorSpace, 30);
    const std::string clientName = camProxy.subscribeCamera("test", 0, kQVGA, kBGRColorSpace, 30);
    //  int fps=camProxy.getFrameRate(0);
    //  cout<<fps<<endl;

    /** Create an cv::Mat header to wrap into an opencv image.*/
    cv::Mat imgHeader = cv::Mat(cv::Size(320, 240), CV_8UC3 );//CV_8UC3);

    /** Create a OpenCV window to display the images. */
    cv::namedWindow("images");

    /* init destin network */
    SupportedImageWidths siw = W256;
    uint centroid_counts[]  = {32,32,32,32,32,16,10};
    bool isUniform = true;
    //bool isUniform = false;
    int nLayers=7;
    DestinNetworkAlt * network = new DestinNetworkAlt(siw, nLayers, centroid_counts, isUniform);
    network->setFixedLearnRate(.1);
    cv::Mat buffer;
    cv::Mat image;

    // belief exporter
    BeliefExporter * featureExtractor=new BeliefExporter(*network, 5);

    /* init destin network end */

    int frameCount=0;

    vector<float> TrainingBeliefs;
    vector<vector<float> > TestingBeliefs;
    vector<float>rowFrame;

    int rowframe=0;                     // a variable to store into different rows each time in TestinBeliefs

    /** Main loop. Exit when pressing ESC.*/
    while ((char) cv::waitKey(30) != 27)
    {
        ALValue img = camProxy.getImageRemote(clientName);
        imgHeader.data = (uchar*) img[6].GetBinary();
        camProxy.releaseImage(clientName);

        frameCount++;

        /* destin processing */

        // convert and resize image
        buffer=imgHeader;
        cv::resize(buffer, buffer, Size(256,256), 0, 0, INTER_LINEAR);
        cv::cvtColor(buffer, image, CV_BGR2GRAY);

        float * float_image=callImage(image);

        float *pbeliefsTrain;
        float *pbeliefsTest;

        int trainingFrame=70;  // number of frames for training

        uint size = featureExtractor->getOutputSize();

        // feed in destin
        network->doDestin(float_image);

        if (frameCount<trainingFrame){
            cout << "[Training] " << frameCount << endl;
        }

        else if(frameCount==trainingFrame)
        {
            // Stop training
            //network->isTraining(false);
            for (int i=0;i<nLayers;i++) network->setLayerIsTraining(i, false);

            // Extract features/belief of training scene
            pbeliefsTrain = featureExtractor->getBeliefs();
            featureExtractor->writeBeliefToMat("TrainingOutput.txt");

            // Assinging beliefs into local vector
            for(int i=0;i<=size;i++){
                TrainingBeliefs.push_back(pbeliefsTrain[i]);
            }
        }

        else if (frameCount>trainingFrame)
        {

            // Extract features of testing scene
            pbeliefsTest = featureExtractor->getBeliefs();
            featureExtractor->writeBeliefToMat("TestingOutput.txt");

            // Testing again assign. Format of storage => TestingBeliefs[frame_number][belief]
            TestingBeliefs.push_back(rowFrame); // create one row for one more frame

            for(int i=0; i<=size; i++){         // populate the created row
                TestingBeliefs[rowframe].push_back(pbeliefsTest[i]);
            }

            // Compare the beliefs for training and testing for computing Euclidean Distance
            float sumCurrent, cumulSum, euclidDist;
            for(int row=0; row<=rowframe; row++){

                for(int i=0; i<=size; i++){
                    sumCurrent=pow( (TrainingBeliefs[i]-TestingBeliefs[row][i]),2);
                    if(i==0)cumulSum=sumCurrent;
                    else cumulSum=cumulSum+sumCurrent;
                }
                euclidDist=sqrt(cumulSum);
            }

            // Calculate similarities of Euclidean Distance. 0 if not similar at all 1 if similar.
            float euclidSim=1/(1+euclidDist);

            cout << euclidSim << endl;

            rowframe++;

            if(euclidSim>0.9){
                // robot stand up
                posture.goToPosture("Stand", 0.5f);
            }
            else
            {
                // robot sit down
                posture.goToPosture("Crouch", 0.5f);
            }


        }



        //        else if(frameCount==trainingFrame)
        //        {
        //            // Stop training
        //            //network->isTraining(false);
        //            for (int i=0;i<nLayers;i++) network->setLayerIsTraining(i, false);

        //            // Extract features/belief of training scene
        //            pbeliefsTrain = featureExtractor->getBeliefs();
        //            featureExtractor->writeBeliefToMat("TrainingOutput.txt");

        //            // Assinging beliefs into local vector
        //            for(int i=0;i<=size;i++){
        //                TrainingBeliefs.push_back(pbeliefsTrain[i]);
        //            }
        //        }

        //        else if (frameCount>trainingFrame)
        //        {

        //            // Extract features of testing scene
        //            pbeliefsTest = featureExtractor->getBeliefs();
        //            featureExtractor->writeBeliefToMat("TestingOutput.txt");

        //            // Assinging beliefs into local vector
        //            for(int i=0;i<=size;i++){
        //                TestFrame.push_back(pbeliefsTest[i]);
        //                TestingBeliefs.push_back(TestFrame);  // TestFrame starts from 0
        //            }

        //            //Compare beliefs which are stored in a multidimentional vector. eg. TestingBeliefs[i][j]. //i is the belief value number j is the testframe

        //            float sum, sumCurrent;
        //            for(int testframe = trainingFrame; testframe<=frameCount ; testframe++){

        //                for(int i=0;i<=size;i++){
        //                    sumCurrent = pow( (TrainingBeliefs[i]-TestingBeliefs[i][testframe]), 2);
        //                    //cout<< "Training Belief " << i <<" : "<< TrainingBeliefs[i] << endl;
        //                    //cout<< "Testing Belief  " << i <<" : "<< TestingBeliefs[i][testframe] << " Testing Frame : " << testframe << endl;
        //                    if (i==0) sum=sumCurrent;
        //                    else sum=sum+sumCurrent;
        //                }
        //                sum=sqrt(sum);
        //                cout << testframe <<" Euclidean Distance: " << sum << endl;

        //            }
        //            // Swapping with an empty dummy vector frees up memory. TestFrame.clear() doesnt.
        //            vector<float>dummyvector;
        //            TestFrame.swap(dummyvector);
        //        }


        //        if (framecount==10)
        //        {
        //            // stop training
        //            network->isTraining(false);
        //            // record feature for current frame using BeliefExporter
        //               // BelifExporter
        //            //featureExtractor->writeBeliefToDisk(1, "result/destin_features/trainOutput.txt");
        //            featureExtractor->writeBeliefToMat("~/trainingOutput.txt");
        //            //cout << "Saved training features" << endl;
        //            cout.flush();

        //        }
        //        else if (framecount>200)  // this will be the testing phase
        //        {
        //            // extract feature here
        //            //featureExtractor->writeBeliefToMat("~/testingOutput.txt");
        //            cout << "Saved testing features" << endl;

        //            // calcluate similarity

        //            // if feature is similar enough
        //            //if(){}
        //            // robot stand up
        //            //else{}
        //            // else robot sit down
        //        }





        // output result

        //        printf("\033[2J\033[1;1H");

        //        printFPS(true);
        //        int layer = 6; // Original: 1
        //        Node & n = *network->getNode(layer,0,0);
        //        printf("Node %i,0,0 winner: %i\n",layer, n.winner);
        //        printf("Node centroids: %i\n", n.nb);

        //        printf("layer %i node 0 centroid locations:\n", layer);
        //        network->printNodeCentroidPositions(layer, 0, 0);
        //        for(int l = 0 ; l < 6 ; l++){
        //            printf("belief graph layer: %i\n",l);
        //            network->printBeliefGraph(l,0,0);
        //        }

        /* destin processing end */


        cv::imshow("images", image);
    }

    /** Cleanup.*/
    delete network;
    camProxy.unsubscribe(clientName);
}

void process2(const std::string & robotIp)
{
    /** Create a proxy to ALVideoDevice on the robot.*/
    ALVideoDeviceProxy camProxy(robotIp, 9559);
    ALRobotPostureProxy posture(robotIp, 9559);

    // sit down as initial postion
    posture.goToPosture("Crouch", 0.5f);

    /** Subscribe a client image requiring 320*240 and BGR colorspace.*/
    //const std::string clientName = camProxy.subscribe("test", kQVGA, kBGRColorSpace, 30);
    const std::string clientName = camProxy.subscribeCamera("test", 0, kQVGA, kBGRColorSpace, 30);
    //  int fps=camProxy.getFrameRate(0);
    //  cout<<fps<<endl;

    /** Create an cv::Mat header to wrap into an opencv image.*/
    cv::Mat imgHeader = cv::Mat(cv::Size(320, 240), CV_8UC3 );//CV_8UC3);

    /** Create a OpenCV window to display the images. */
    cv::namedWindow("Scene");

    /* init destin network */
    SupportedImageWidths siw = W256;
    uint centroid_counts[]  = {32,32,32,32,32,16,10};
    bool isUniform = true;
    //bool isUniform = false;
    int nLayers=7;
    DestinNetworkAlt * network = new DestinNetworkAlt(siw, nLayers, centroid_counts, isUniform);
    network->setFixedLearnRate(.1);
    cv::Mat buffer;
    cv::Mat image;

    // belief exporter
    BeliefExporter * featureExtractor=new BeliefExporter(*network, 5);

    /* init destin network end */

    int frameCount=0;

    float *pbeliefsTrain;
    float *pbeliefsTest;

    int numTrainingFrame=70;  // number of frames for training

    uint size = featureExtractor->getOutputSize();

    // array init
    float testArray[size];
    float trainArray[size];
    network->clearBeliefs();//clear network before start

    /** Main loop. Exit when pressing ESC.*/
    while ((char) cv::waitKey(30) != 27)
    {
        ALValue img = camProxy.getImageRemote(clientName);
        imgHeader.data = (uchar*) img[6].GetBinary();
        camProxy.releaseImage(clientName);

        // convert and resize image
        buffer=imgHeader;
        cv::resize(buffer, buffer, Size(256,256), 0, 0, INTER_LINEAR);
        cv::cvtColor(buffer, image, CV_BGR2GRAY);

        float * float_image=callImage(image);

        // feed in destin
        network->doDestin(float_image);

        if (frameCount<numTrainingFrame){
            cout << "[Training] " << frameCount << endl;
        }

        else if(frameCount==numTrainingFrame)
        {
            // Stop training
            network->isTraining(false);

            // Extract features/belief of training scene
            pbeliefsTrain = featureExtractor->getBeliefs();

            // Assinging beliefs into local array
            for(int i=0;i<=size;i++){
                trainArray[i]=pbeliefsTrain[i];
            }
        }

        else if (frameCount>numTrainingFrame)
        {
            // Extract features of testing scene
            pbeliefsTest = featureExtractor->getBeliefs();

            for(int i=0; i<=size; i++){
                testArray[i]=pbeliefsTest[i];
            }

            // Compare the beliefs for training and testing for computing Euclidean Distance
            float sumCurrent, cumulSum, euclidDist;
            for(int i=0; i<=size; i++){
                sumCurrent=pow( (trainArray[i]-testArray[i]),2);
                if(i==0)cumulSum=sumCurrent;
                else cumulSum=cumulSum+sumCurrent;
            }
            euclidDist=sqrt(cumulSum);

            // Calculate similarities of Euclidean Distance. 0 if not similar at all 1 if similar.
            float euclidSim=1/(1+euclidDist);
            cout << euclidSim << endl;

            if(euclidSim>0.9){
                // robot stand up
                posture.goToPosture("Stand", 0.5f);
            }
            else
            {
                // robot sit down
                posture.goToPosture("Crouch", 0.5f);
            }
        }
        frameCount++;
        cv::imshow("Scene", image);
    }

    /** Cleanup.*/
    delete network;
    camProxy.unsubscribe(clientName);
}

void processWebcam()
{
    //VideoSource vs(false, "./Various.avi");
    VideoSource vs(true, "");
    vs.enableDisplayWindow();
    SupportedImageWidths siw = W256;

    // Left to Right is bottom layer to top
    uint centroid_counts[]  = {32,32,32,32,32,16,10};
    bool isUniform = true;
    int nLayers=7;
    DestinNetworkAlt * network = new DestinNetworkAlt(siw, nLayers, centroid_counts, isUniform);
    network->setFixedLearnRate(.1);
    BeliefExporter * featureExtractor=new BeliefExporter(*network, 5);

    Transporter t;
    vs.grab();//throw away first frame in case its garbage
    int frameCount = 0;

    vector<float> TrainingBeliefs;          // vector for storing the Training scene
    vector<vector<float> > TestingBeliefs;  // create 2D vector for storing the features of testing scene
    vector<float>rowFrame;

    int rowframe=0;                         // a variable to store into different rows each time in TestingBeliefs

    while(vs.grab()){

        frameCount++;

        t.setSource(vs.getOutput());
        t.transport();                      //move video from host to card
        testNan(t.getDest(), 512*512);

        float *pbeliefsTrain;
        float *pbeliefsTest;

        int trainingFrame= 60;              //70-90 is ok // number of frames for training
        uint size = featureExtractor->getOutputSize();

        //feed the video into the destin algorithm
        network->doDestin(t.getDest());

        if (frameCount<trainingFrame){
            cout << "[Training] " << frameCount << endl;
        }

        else if(frameCount==trainingFrame)
        {
            // Stop training
            network->isTraining(false);
            //for (int i=0;i<nLayers;i++) network->setLayerIsTraining(i, false);

            // Extract features/belief of training scene
            pbeliefsTrain = featureExtractor->getBeliefs();
            featureExtractor->writeBeliefToMat("TrainingOutput.txt");

            // Assinging beliefs into local vector
            for(int i=0;i<=size;i++){
                TrainingBeliefs.push_back(pbeliefsTrain[i]);
            }
        }

        else if (frameCount>trainingFrame)
        {

            // Extract features of testing scene
            pbeliefsTest = featureExtractor->getBeliefs();
            featureExtractor->writeBeliefToMat("TestingOutput.txt");

            // OLD ASSIGN BELIEF
            //            // Assinging beliefs into local vector
            //            for(int i=0;i<=size;i++){
            //                TestFrame.push_back(pbeliefsTest[i]); // push belief into vector
            //                //TestingBeliefs.push_back(TestFrame);  // testframe starts from 0  // Format ==> TestingBeliefs[i][testframe]
            //            }

            // Testing again assign. Format of storage => TestingBeliefs[frame_number][belief]
            TestingBeliefs.push_back(rowFrame); // create one row for one more frame

            for(int i=0; i<=size; i++){         // populate the created row
                TestingBeliefs[rowframe].push_back(pbeliefsTest[i]);
            }

            // Test to print the variables
            //            for(int frame=0; frame<=rowframe; frame++){
            //                for(int i=0; i<=size; i++){
            //                    cout << "TestingBeliefs["<< frame << "]"<<"["<<i<<"]: "<< TestingBeliefs[frame][i]<<endl;
            //                }
            //            }


            // Compare the beliefs for training and testing for computing Euclidean Distance
            float sumCurrent, cumulSum, euclidDist;
            for(int row=0; row<=rowframe; row++){

                for(int i=0; i<=size; i++){
                    sumCurrent=pow( (TrainingBeliefs[i]-TestingBeliefs[row][i]),2);
                    if(i==0)cumulSum=sumCurrent;
                    else cumulSum=cumulSum+sumCurrent;
                }
                euclidDist=sqrt(cumulSum);
            }

            // Calculate similarities of Euclidean Distance. 0 if not similar at all 1 if similar.
            float euclidSim=1/(1+euclidDist);

            cout << euclidSim << endl;

            rowframe++;



            // OLD COMPARE
            // OLD Compare beliefs which are stored in a multidimentional vector. eg. TestingBeliefs[i][j].
            // i is the belief value number j is the testframe
            //            float sum, sumCurrent;
            //            for(int testingframe = trainingFrame; testingframe<frameCount ; testingframe++){

            //                for(int i=0;i<=size;i++){
            //                    sumCurrent = pow( (TrainingBeliefs[i]-TestingBeliefs[i][testingframe]), 2);
            //                    //cout<< "Training Belief " << i <<" : "<< TrainingBeliefs[i] << endl;
            //                    //cout<< "Testing Belief  " << i <<" : "<< TestingBeliefs[i][testframe] << " Testing Frame : " << testframe << endl;
            //                    if (i==0) sum=sumCurrent;
            //                    else sum=sum+sumCurrent;
            //                }
            //                // This is the computer Eucliden Distance for ONE frame.
            //                double euclidDist=sqrt(sum);

            //                // Calculate the euclidean similarities. Returns 0 if not similar , 1 if similar
            //                double euclidSimilar=1/(1+euclidDist);
            //                cout << euclidSimilar<< endl;

            ////                cout <<"Test frame number: "<< testframe <<" Euclidean Distance: " << euclidMeanSum << endl;
            //            }

            // may be good to erase some data from vector to free some memory. testing. Seems to be working.
            // Swapping with an empty dummy vector frees up memory. TestFrame.clear() doesnt.
            // vector<float>dummyvector;
            //TestFrame.swap(dummyvector);
            //euclidArr.swap(dummyvector);
            //TestingBeliefs.swap(dummyvector2);

        }

    }
    delete network;
}


void processWebcam2()  // implementation without vectors but array
{
    //VideoSource vs(false, "./Various.avi");
    VideoSource vs(true, "");
    vs.enableDisplayWindow();
    SupportedImageWidths siw = W256;

    // Left to Right is bottom layer to top
    uint centroid_counts[]  = {32,32,32,32,32,16,10};
    bool isUniform = true;
    int nLayers=7;
    DestinNetworkAlt * network = new DestinNetworkAlt(siw, nLayers, centroid_counts, isUniform);
    network->setFixedLearnRate(.1);
    BeliefExporter * featureExtractor=new BeliefExporter(*network, 5);

    Transporter t;
    vs.grab();//throw away first frame in case its garbage
    int frameCount = 0;

    float *pbeliefsTrain;
    float *pbeliefsTest;
    int numTrainingFrame= 70;              //70-90 is ok // number of frames for training
    uint size = featureExtractor->getOutputSize();

    // array init
    float testArray[size];
    float trainArray[size];
    network->clearBeliefs();//clear network before start

    while(vs.grab()){

        t.setSource(vs.getOutput());
        t.transport();                      //move video from host to card
        testNan(t.getDest(), 512*512);

        //feed the video into the destin algorithm
        network->doDestin(t.getDest());

        if (frameCount<numTrainingFrame){
            cout << "[Training] " << frameCount << endl;
        }

        else if(frameCount==numTrainingFrame)
        {
            // Stop training
            network->isTraining(false);

            // Extract features/belief of training scene
            pbeliefsTrain = featureExtractor->getBeliefs();

            // Assinging beliefs into local array
            for(int i=0;i<=size;i++){
                trainArray[i]=pbeliefsTrain[i];
            }
        }

        else if (frameCount>numTrainingFrame)
        {
            // Extract features of testing scene
            pbeliefsTest = featureExtractor->getBeliefs();

            for(int i=0; i<=size; i++){
                testArray[i]=pbeliefsTest[i];
            }

            // Compare the beliefs for training and testing for computing Euclidean Distance
            float sumCurrent, cumulSum, euclidDist;
            for(int i=0; i<=size; i++){
                sumCurrent=pow( (trainArray[i]-testArray[i]),2);
                if(i==0)cumulSum=sumCurrent;
                else cumulSum=cumulSum+sumCurrent;
            }
            euclidDist=sqrt(cumulSum);

            // Calculate similarities of Euclidean Distance. 0 if not similar at all 1 if similar.
            float euclidSim=1/(1+euclidDist);
            cout << euclidSim << endl;
        }
        frameCount++;
    }
    delete network;
}

void showImages(const std::string& robotIp){
    /** Create a proxy to ALVideoDevice on the robot.*/
    ALVideoDeviceProxy camProxy(robotIp, 9559);

    /** Subscribe a client image requiring 320*240 and BGR colorspace.*/
    //const std::string clientName = camProxy.subscribe("test", kQVGA, kBGRColorSpace, 30);
    const std::string clientName = camProxy.subscribeCamera("test", 0, kQVGA, kBGRColorSpace, 30);
    //  int fps=camProxy.getFrameRate(0);
    //  cout<<fps<<endl;

    /** Create an cv::Mat header to wrap into an opencv image.*/
    cv::Mat imgHeader = cv::Mat(cv::Size(320, 240), CV_8UC3 );//CV_8UC3);

    /** Create a OpenCV window to display the images. */
    cv::namedWindow("images");

    /** Main loop. Exit when pressing ESC.*/
    while ((char) cv::waitKey(30) != 27)
    {
        /** Retrieve an image from the camera.
    * The image is returned in the form of a container object, with the
    * following fields:
    * 0 = width
    * 1 = height
    * 2 = number of layers
    * 3 = colors space index (see alvisiondefinitions.h)
    * 4 = time stamp (seconds)
    * 5 = time stamp (micro seconds)
    * 6 = image buffer (size of width * height * number of layers)
    */
        ALValue img = camProxy.getImageRemote(clientName);

        /** Access the image buffer (6th field) and assign it to the opencv image
    * container. */
        imgHeader.data = (uchar*) img[6].GetBinary();

        /** Tells to ALVideoDevice that it can give back the image buffer to the
    * driver. Optional after a getImageRemote but MANDATORY after a getImageLocal.*/
        camProxy.releaseImage(clientName);

        /** Display the iplImage on screen.*/
        cv::imshow("images", imgHeader);
    }

    /** Cleanup.*/
    camProxy.unsubscribe(clientName);
}


//***********************************Main loop*******************************************


int main(int argc, char ** argv)
{

    try
    {
        process2(robotIp);
        //processWebcam2();
    }
    catch (const AL::ALError& e)
    {
        std::cerr << "Caught exception " << e.what() << std::endl;
    }


    return 0;
}


