//Ayobami Ige
//Kurt Palo

#include <iostream>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "math.h"


using namespace cv;
using namespace std;

//Global variables
string faceCascade = "/home/kurt/ClionProjects/Vision/haarcascade_frontalface_alt.xml";
string eyeCascade = "/home/kurt/ClionProjects/Vision/haarcascade_eye_tree_eyeglasses.xml";
string noseCascade = "/home/kurt/ClionProjects/Vision/haarcascade_mcs_nose.xml";
string mouthCascade= "/home/kurt/ClionProjects/Vision/haarcascade_mcs_mouth.xml";


//Cascade classifiers
CascadeClassifier face, eyes, nose, mouth;

//Mat variables
Mat source,source2, output1, output2;

double distance(Point first, Point second){

    return sqrt(pow(second.x - first.x, 2) + pow(second.y - first.y, 2));

}

//Draw triangulating lines
double drawLines(vector<Rect>, vector<Rect>,vector<Rect>,vector<Rect>,Mat);
//Draw triangulating lines
double drawLines(vector<Rect> face1, vector<Rect> eyes, vector<Rect>mouth, vector<Rect> nose, Mat output){

    double eye1ToEye2=0.0;
    double eye1ToNose = 0.0;
    double eye1ToMouth=0.0;
    double eye2ToNose = 0.0;
    double eye2ToMouth=0.0;
    double noseToMouth=0.0;

    //Draw lines
    //Eye 1 to Eye 2
    Point centerLeft = Point(face1[0].x + eyes[0].x + eyes[0].width / 2,
                             face1[0].y + eyes[0].y + eyes[0].height / 2);
    Point centerRight =  Point(face1[0].x + eyes[1].x + eyes[1].width / 2,
                               face1[0].y + eyes[1].y + eyes[1].height / 2);
    line(output, centerLeft, centerRight, Scalar(0,0,0), 2);

    //Nose center point
    Point noseCenter = Point(face1[0].x + nose[0].x + nose[0].width / 2,
                             face1[0].y + nose[0].y + nose[0].height / 2);

    //Eye 1 to nose
    line(output, centerLeft, noseCenter, Scalar(0, 0, 0), 2);

    //Eye 2 to nose
    line(output, centerRight, noseCenter, Scalar(0, 0, 0), 2);

    //Nose to mouth
    Point mouthCenter  = Point(face1[0].x + mouth[0].x + mouth[0].width / 2,
                               face1[0].y + mouth[0].y + mouth[0].height / 2);
    line(output, noseCenter, mouthCenter, Scalar(0, 0, 0), 2);

    //Mouth center to eye 1
    line(output, mouthCenter, centerLeft, Scalar(0, 0, 0), 2);

    //Mouth center to eye 2
    line(output, mouthCenter, centerRight, Scalar(0, 0, 0), 2);

    //Get the distance of the points

    //Length of vector from eye 1 to eye 2
    eye1ToEye2 = distance(centerLeft, centerRight);

    //Eye 1 to nose
    eye1ToNose = distance(centerLeft, noseCenter);

    //Eye 1 to mouth
    eye1ToMouth = distance(centerLeft, mouthCenter);

    //eye 2 to mouth
    eye2ToMouth = distance(centerRight, mouthCenter);

    //Eye 2 to nose
    eye2ToNose =distance(centerRight, noseCenter);

    //Nose to mouth
    noseToMouth = distance(noseCenter, mouthCenter);

    double lengths = eye1ToEye2 + eye1ToNose + eye2ToNose + noseToMouth + eye1ToMouth + eye2ToMouth;
    return lengths;
}


//Detect a face from the loaded Mat and return a vector
vector<Rect> detectFace(Mat frame){

    vector<Rect> faceVec;

    //Checks if cascade files where loaded
    if(!face.load(faceCascade)) {
        cout << "Error loading face cascade" << endl;
        return faceVec;
    }

    //Detect the features and store it into vector face
    face.detectMultiScale(frame, faceVec, 1.1, 6, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10));

    return faceVec;
}

//Detect nose from the Region of Interest
vector<Rect> detectNose(Mat frame){

    vector<Rect> noseVec;

    //Check if nose cascade can be loaded
    if(!nose.load(noseCascade)) {
        cout << "Error loading nose cascade" << endl;
        return noseVec;
    }

    nose.detectMultiScale(frame, noseVec, 1.1, 6, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10));

    return noseVec;
}


//Detect nose from the Region of Interest
vector<Rect> detectMouth(Mat frame){

    vector<Rect> mouthVec;

    //Check if nose cascade can be loaded
    if(!mouth.load(mouthCascade)) {
        cout << "Error loading mouth cascade" << endl;
        return mouthVec;
    }

    mouth.detectMultiScale(frame, mouthVec, 1.1, 6, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10));

    return mouthVec;
}
//Detect the eyes from the Region of Interest
vector<Rect> detectEyes(Mat frame){

    vector<Rect> eyeVec;

    // checks if cascade files are loaded
    if(!eyes.load(eyeCascade)) {
        cout << "Error loading eye cascade" << endl;
        return eyeVec;
    }

    eyes.detectMultiScale(frame, eyeVec, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, Size(15, 15));

    return eyeVec;
}

//Draw function for drawing the eyes and face
double drawFace(Mat frame, Mat output, string name) {

    //Convert to gray scale
    Mat gray;
    cvtColor(frame, gray, CV_BGR2GRAY);

    frame.copyTo(output);
    //Vectors for eyes and face detection
    vector<Rect> faceVector, eyesVector, mouthVector, noseVector;


    //Detect the face and eye features then store them into vectors
    faceVector = detectFace(frame);

    //Draw the face shape
    for (int i = 0; i < faceVector.size(); i++) {

        //Draw an ellipse around the face
        Point center(faceVector[i].x + faceVector[i].width / 2, faceVector[i].y + faceVector[i].height / 2);
        ellipse(output, center, Size(faceVector[i].width / 2, faceVector[i].height / 2), 0, 0, 360, Scalar(0, 0, 255),
                2);

        //Create a Region of Interest of the face
        Mat faceROI = gray(faceVector[i]);

        //Detect eyes
        eyesVector = detectEyes(faceROI);

        //Draw the eye shapes
        for (int j = 0; j < eyesVector.size(); j++) {

            //Draw the ellipse around the eyes
            Point center(faceVector[i].x + eyesVector[j].x + eyesVector[j].width / 2,
                         faceVector[i].y + eyesVector[j].y + eyesVector[j].height / 2);
            int radius = cvRound((eyesVector[j].width + eyesVector[j].height) * .25);
            circle(output, center, radius, Scalar(255, 0, 0), 2);
        }

        //Draw nose cascade
        double noseHeight = 0.0;
        noseVector = detectNose(faceROI);
        for (int k = 0; k < noseVector.size(); k++) {

            //Draw the ellipse around the nose
            Point center(faceVector[i].x + noseVector[k].x + noseVector[k].width / 2,
                         faceVector[i].y + noseVector[k].y + noseVector[k].height / 2);
            int radius = cvRound((noseVector[k].width + noseVector[k].height) * .05);
            circle(output, center, radius, Scalar(0, 255, 0), 2);
            noseHeight = noseVector[k].y + noseVector[k].height / 2;

        }


        //Draw mouth cascade
        double mouth_center_height = 0.0;
        mouthVector = detectMouth(faceROI);

        for (int p = 0; p < mouthVector.size(); p++) {

            //Draw the rectangle around the mouth
            Rect m = mouthVector[p];
            mouth_center_height = m.y;

            // The mouth should lie below the nose
            if (mouth_center_height > noseHeight) {

                rectangle(output, Point(faceVector[i].x + mouthVector[p].x,
                                        faceVector[i].y + mouthVector[p].y),
                          Point(faceVector[i].x + mouthVector[p].x + mouthVector[p].width,
                                faceVector[i].y + mouthVector[p].y + mouthVector[p].height), Scalar(155, 255, 120), 2, 4);

            } else if (mouth_center_height <= noseHeight) {
                continue;
            }
        }


        double img1 = drawLines(faceVector, eyesVector, mouthVector, noseVector, output);

        namedWindow(name, CV_WINDOW_AUTOSIZE);// Create a window for display.
        imshow(name, output);

        return img1;
    }
}

int main(int argc, char* argv[]) {
double  final = 0.0;
    while(1) {

        //face 1 and face 11

        //Load the image onto source
        source = imread("/home/kurt/ClionProjects/Vision/obama1.jpg");
        source2 = imread("/home/kurt/ClionProjects/Vision/face11.jpg");

        //Copy source image into output/

        double x = drawFace(source,output1,"img1") / 6;
        double y = drawFace(source2,output2,"img2") / 6;
        if(x> y)

        final = (x - y) * 100 ;
        else
            final = (y-x) * 100;
        cout << final << endl;

        // Show our image inside it.
        if (waitKey(30) == 27) {

            destroyAllWindows();
            break;
        }
    }

    return 0;
}