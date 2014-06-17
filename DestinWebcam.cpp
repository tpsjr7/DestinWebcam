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

using namespace AL;
using namespace std;
using namespace cv;

const std::string robotIp="192.168.0.109";

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

    int framecount=0;

    /** Main loop. Exit when pressing ESC.*/
    while ((char) cv::waitKey(30) != 27)
    {
        ALValue img = camProxy.getImageRemote(clientName);
        imgHeader.data = (uchar*) img[6].GetBinary();
        camProxy.releaseImage(clientName);

        /* destin processing */

        // convert and resize image
        buffer=imgHeader;
        cv::resize(buffer, buffer, Size(256,256), 0, 0, INTER_LINEAR);
        cv::cvtColor(buffer, image, CV_BGR2GRAY);

        float * float_image=callImage(image);

        // feed in destin
        network->doDestin(float_image);

        if (framecount==10)
        {
            // stop training
            network->isTraining(false);
            // record feature for current frame using BeliefExporter
               // BelifExporter
            //featureExtractor->writeBeliefToDisk(1, "result/destin_features/trainOutput.txt");
            featureExtractor->writeBeliefToMat("~/trainingOutput.txt");
            //cout << "Saved training features" << endl;
            cout.flush();

        }
        else if (framecount>200)  // this will be the testing phase
        {
            // extract feature here
            //featureExtractor->writeBeliefToMat("~/testingOutput.txt");
            cout << "Saved testing features" << endl;

            // calcluate similarity

            // if feature is similar enough
            //if(){}
            // robot stand up
            //else{}
            // else robot sit down
        }





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

        cout<<"Framecount: "<<framecount<<endl;
        framecount++;
    }

    /** Cleanup.*/
    delete network;
    camProxy.unsubscribe(clientName);
}

void processWebcam()
{
    //VideoSource vs(false, "/home/dickson/Desktop/destin/Destin/Misc/./Various.avi");
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

    while(vs.grab()){

        frameCount++;

        t.setSource(vs.getOutput());
        t.transport(); //move video from host to card
        testNan(t.getDest(), 512*512);

        float *beliefsTrain;
        float *beliefsTest;
        uint size = featureExtractor->getOutputSize();

        network->doDestin(t.getDest());
        int frame=1000;

        if (frameCount<frame){cout << frameCount << endl;}

        else if(frameCount==frame)
        {
            // Stop training
            network->isTraining(false);
            for (int i=0;i<nLayers;i++) network->setLayerIsTraining(i, false);

            // Extract features/belief of training scene
            beliefsTrain = featureExtractor->getBeliefs();
            featureExtractor->writeBeliefToMat("TrainingOutput.txt");

//            // Printing the beliefs
//            for(int i=0;i<=size;i++){
//                cout << beliefsTrain[i]<< endl;
//            }

        }

        else if (frameCount>frame)
        {

            // Extract features of testing scene
            beliefsTest = featureExtractor->getBeliefs();
            featureExtractor->writeBeliefToMat("TestingOutput.txt");

            // Compare beliefs
            int sum;
            for(int i=0;i<=size;i++){
                sum = pow( (beliefsTest[i]-beliefsTrain[i]), 2);
                cout<< "Training Belief " << i <<" : "<< beliefsTrain[i]<< endl;
                cout<< "Testing Belief  " << i <<" : "<< beliefsTest[i]<< endl;
            }
            sum=sqrt(sum);
            cout <<"Euclidean Distance: " << sum << endl;



             // Printing the beliefs
//            for(int i=0;i<=size;i++){
//                cout << beliefsTest[i]<< endl;
//            }
        }

//        printf("\033[2J\033[1;1H");

//        printf("Frame: %i\n", frameCount);
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

    }
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
//    cv::Mat mat;
//    mat=cv::imread("/home/dickson/img.jpg");
//    cvNamedWindow("OpenCV in QT");
//    cv:imshow("OpenCV in QT", mat);
//    cvWaitKey(5000);

//    AL::ALTextToSpeechProxy tts(robotIp, 9559);
//    const std::string phraseToSay = "";
//    tts.say(phraseToSay);

    try
    {
        //process(robotIp);
        processWebcam();
    }
    catch (const AL::ALError& e)
    {
        std::cerr << "Caught exception " << e.what() << std::endl;
    }

    return 0;
}


