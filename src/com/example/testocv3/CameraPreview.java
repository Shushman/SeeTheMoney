//This is a standard class for handing the preview of the camera feed

package com.example.testocv3;

import java.io.IOException;

import android.content.Context;
import android.hardware.Camera;
import android.hardware.Camera.AutoFocusCallback;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

public class CameraPreview extends SurfaceView implements SurfaceHolder.Callback {
	
    private SurfaceHolder mSurfaceHolder;
    private Camera mCamera;
    private static String TAG = "See_The_Money.CameraPreview";
 
    //Constructor that obtains context and camera
    public CameraPreview(Context context, Camera camera) {
        super(context);
        Log.d(TAG,"Inside CameraPreview constructor");
        this.mCamera = camera;
        this.mSurfaceHolder = this.getHolder();
        this.mSurfaceHolder.addCallback(this); // we get notified when underlying surface is created and destroyed
        this.mSurfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS); //this is a deprecated method, is not requierd after 3.0
    }
 
    @Override
    public void surfaceCreated(SurfaceHolder surfaceHolder) {
    	Log.d(TAG,"Inside surfaceCreated function");
        try {
        	//
            mCamera.setPreviewDisplay(surfaceHolder);
            mCamera.startPreview();
            mCamera.autoFocus(autoFocusCallback);
        } catch (IOException e) {

        }
 
    }
     
    @Override
    public void surfaceDestroyed(SurfaceHolder surfaceHolder) {
    	Log.d(TAG,"Inside surfaceDestroyed function");
        mCamera.stopPreview();
        mCamera.release();
    }
 
    @Override
    public void surfaceChanged(SurfaceHolder surfaceHolder, int format,
            int width, int height) {
        // start preview with new settings
    	Log.d(TAG,"Inside surfaceChanged function");
        try {
            mCamera.setPreviewDisplay(surfaceHolder);
            mCamera.startPreview();
            mCamera.autoFocus(autoFocusCallback);
        } catch (Exception e) {
            // intentionally left blank for a test
        }
    }
    
    AutoFocusCallback autoFocusCallback = new AutoFocusCallback() {
  	  @Override
  	  public void onAutoFocus(boolean success, Camera camera) {
  	    Log.i(TAG,"Camera focussed"); 
  	   
  	  }
  	};
     
}