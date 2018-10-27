package com.example.rm.opencvexample;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.InstallCallbackInterface;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * https://medium.com/@aashari/simple-rectangle-detection-using-opencv-on-android-48e2a9a0586a
 */

public class OpenCVCamera extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "OpenCVCamera";
    // View Holder
    private CameraBridgeViewBase cameraBridgeViewBase;
    // Camera Listener Callback
    private BaseLoaderCallback baseLoaderCallback;
    // Image holder
    private Mat bwIMG;
    private Mat hsvIMG;
    private Mat lrrIMG;
    private Mat urrIMG;
    private Mat dsIMG;
    private Mat usIMG;
    private Mat cIMG;
    private Mat hovIMG;

    //These use variables are used to fix camera orientation
    Mat mRgba;
    Mat mRgbaF;
    Mat mRgbaT;

    Mat mGray;
    Mat mGrayF;
    Mat mGrayT;

    private MatOfPoint2f approxCurve;

    private int threshold;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_opencv_camera);

        //Initialize threshold
        threshold = 100;

        cameraBridgeViewBase = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);

        // Create Camera listener CallBack
        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS:
                        Log.v("OpenCvCamer", "Load Success");
                        bwIMG = new Mat();
                        dsIMG = new Mat();
                        hsvIMG = new Mat();
                        lrrIMG = new Mat();
                        urrIMG = new Mat();
                        usIMG = new Mat();
                        cIMG = new Mat();
                        hovIMG = new Mat();
                        approxCurve = new MatOfPoint2f();
                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }
            }

            @Override
            public void onPackageInstall(int operation, InstallCallbackInterface callback) {
                super.onPackageInstall(operation, callback);
            }
        };
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        // Receive Image Data when the camera preview starts on your screen 😀

        //CvType.CV_8UC4 ?:
        //This types define color depth, number of channels and channel layout in the image.
        // On Android the most useful are CvType.CV_8UC4 and CvType.CV_8UCq.
        // CvType.CV_8UC4 is 8-bit per channel RGBA image and can be captured from camera with
        // NativeCameraView or JavaCameraView classes and drawn on surface.
        // CvType.CV_8UC1 is gray scale image and is mostly used in computer vision algorithms.
        mRgba = new Mat(height,width,CvType.CV_8UC4);
        mRgbaF = new Mat(height,width,CvType.CV_8UC4);
        mRgbaT = new Mat(height,width,CvType.CV_8UC4);

        mGray = new Mat(height,width,CvType.CV_8UC1);
        mGrayF = new Mat(height,width,CvType.CV_8UC1);
        mGrayT = new Mat(height,width,CvType.CV_8UC1);

    }

    @Override
    public void onCameraViewStopped() {
        //Destroy image data when you stop camera preview on your phone screen
        mRgba.release();

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {


        // Now, this one is interesting! OpenCV orients the camera to left by 90 degrees.
        // So if the app is in portrait more, camera will be in -90 or 270 degrees orientation.
        // We fix that in the next and the most important function. There you go!
        mRgba = inputFrame.rgba();
        //Core.transpose(mRgba,mRgbaT);
        //Imgproc.resize(mRgbaT,mRgbaF,mRgbaF.size(),0,0,0);
        //Core.flip(mRgbaF,mRgba,1);

        mGray = inputFrame.gray();
        //Core.transpose(mGray, mGrayF);
        //Imgproc.resize(mGrayT,mGrayF,mGrayF.size(),0,0,0);
        //Core.flip(mGrayF,mGray,1);

        // Reduce image size by 50% and save int dsIMG
        // https://www.tutorialspoint.com/opencv/opencv_image_pyramids.htm
        Imgproc.pyrDown(mGray, dsIMG, new Size(mGray.cols() / 2, mGray.rows() / 2));

        //  Increases the image size and sabe in usIMG
        // https://www.tutorialspoint.com/opencv/opencv_image_pyramids.htm
        Imgproc.pyrUp(dsIMG, usIMG, mGray.size());

        // Canny Edge Detection is used to detect the edges in an image. and save value in bwIMG.
        // https://www.tutorialspoint.com/opencv/opencv_canny_edge_detection.htm
        Imgproc.Canny(usIMG, bwIMG, 0, threshold);

        // It dilates an image by using a specific structuring element.
        // https://www.tutorialspoint.com/java_dip/eroding_dilating.htm
        Imgproc.dilate(bwIMG, bwIMG, new Mat(), new Point(-1, 1), 1);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

        cIMG = bwIMG.clone();

        // Find contours in binary image
        Imgproc.findContours(cIMG, contours, hovIMG, Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint cnt : contours) {
            MatOfPoint2f curve = new MatOfPoint2f(cnt.toArray());

            // Approximates a polygonal curve(s) with the specified precision.
            Imgproc.approxPolyDP(curve, approxCurve, 0.02 *
                    Imgproc.arcLength(curve, true), true);

            int numberVertices = (int) approxCurve.total();

            double contourArea = Imgproc.contourArea(cnt);

            if (Math.abs(contourArea) < 100) {
                continue;
            }

            //Rectangle detected
            if (numberVertices >= 4 && numberVertices <= 6) {

                List<Double> cos = new ArrayList<>();

                for(int j = 2; j < numberVertices + 1; j++){
                    cos.add(
                            angle(
                                    approxCurve.toArray()[j %  numberVertices],
                                    approxCurve.toArray()[j-2],
                                    approxCurve.toArray()[j-1]));
                }

                Collections.sort(cos);

                double mincos = cos.get(0);
                double maxcos = cos.get(cos.size() - 1);

                if(numberVertices == 4 && mincos >= -0.1 && maxcos <= 3){
                    setLabel(mRgba, "X", cnt);
                }

            }
        }

        return mRgba;

    }

    private static double angle(Point pt1, Point pt2, Point pt0) {
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return (dx1 * dx2 + dy1 * dy2) / Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
    }

    private void setLabel(Mat im, String label, MatOfPoint contour) {
        int fontface = Core.FONT_HERSHEY_SIMPLEX;
        double scale = 3;//0.4;
        int thickness = 3;//1;
        int[] baseline = new int[1];
        Size text = Imgproc.getTextSize(label, fontface, scale, thickness, baseline);
        Rect r = Imgproc.boundingRect(contour);
        Point pt = new Point(r.x + ((r.width - text.width) / 2),r.y + ((r.height + text.height) / 2));
        Imgproc.putText(im, label, pt, fontface, scale, new Scalar(255, 0, 0), thickness);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (cameraBridgeViewBase != null)
            cameraBridgeViewBase.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, baseLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase != null)
            cameraBridgeViewBase.disableView();
    }


}