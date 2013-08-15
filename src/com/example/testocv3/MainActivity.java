package com.example.testocv3;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.hardware.Camera;
import android.hardware.Camera.AutoFocusCallback;
import android.hardware.Camera.PictureCallback;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.AudioManager;
import android.media.AudioManager.OnAudioFocusChangeListener;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.OnTouchListener;
import android.widget.FrameLayout;


public class MainActivity extends Activity implements SensorEventListener{

	private static Camera mCamera;//The variable for controlling the camera
	private static CameraPreview mCameraPreview;//The variable for the preview of camera feed
	private FrameLayout preview;//The preview layout
	private SensorManager mSensorManager;//For managing the accelerometer
	private Sensor mSensor;//The variable for accessing the accelerometer
	private MediaPlayer mplayer;//Controls the audio player
	private static int limit = 30;
	private static int count=limit;//For the number of seconds the user must hold the phone still
	private Resources resources;//For storing the yml file the first time  
	private float oldvals[] = new float[3];
	private float gravity[] = new float[3];
	private float lin_acc[] = new float[3];
	private static float choices[][] = new float[6][2];//For evaluating results of matching
	private static int cycleCount = 0;//Required when multiple images of the same note are taken
	private final float HI_THRESH = 4.2f;//Upper threshold for shaking the phone
	private final float LOW_THRESH = 1.6f;//Lower threshold for shaking the phone
	private static boolean flag = true;//Required to ensure changes of state while shaking
	public static float values[] = new float[3];
	public static File pictureFile;//To refer to each pic taken
	public static String fpath;//The path of each pic taken
	public static String vocabpath;//The path to the trained matrix
	public static final String TAG = "See_The_Money.MainActivity";
	private float val = -1;

	private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		//For loading OpenCV libraries correctly
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS:
			{
				Log.i(TAG, "OpenCV loaded successfully");
				// Load native library after(!) OpenCV initialization
				System.loadLibrary("test_op3");     
			} break;
			default:
			{
				super.onManagerConnected(status);
			} break;
			}
		}
	};

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		Log.i(TAG, "OnCreate function called");
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		resources = getResources();
		saveYmlFile();//Check and store the yml vocabulary file
		
		//Setup the camera and preview
		if(!checkCameraHardware()) finish(); 
		mCamera = getCameraInstance();
		Camera.Parameters cp = mCamera.getParameters();
		cp.setPictureSize(1280,960);
		mCamera.setParameters(cp);
		mCameraPreview = new CameraPreview(this, mCamera);
		preview = (FrameLayout)findViewById(R.id.camera_preview);
		preview.addView(mCameraPreview);
		count = limit;
		
		//Setup the accelerometer and audio player
		mSensorManager = (SensorManager) getSystemService(this.SENSOR_SERVICE);
		mSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
		AudioManager audioManager = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
		int result = audioManager.requestAudioFocus(afChangeListener, AudioManager.STREAM_MUSIC,AudioManager.AUDIOFOCUS_GAIN);

		if (result != AudioManager.AUDIOFOCUS_REQUEST_GRANTED) {
			finish();
		}
		preview.setOnTouchListener(new OnTouchListener() {
			@Override
			public boolean onTouch(View v, MotionEvent event) {
				Log.d(TAG,"Screen has been touched");
				finish();
				return true;
			}
		});
	}

	private void saveYmlFile(){
		//Stores vocabulary file in internal memory if it does not exist
		File ymlpath = getBaseContext().getFileStreamPath("orb4k_p15e15.yml");
		vocabpath = ymlpath.getPath();
		
		if(ymlpath.exists())//File exists and does not need to be saved again
			return;
		else{
			Log.d(TAG,"Training matrix is saved at"+ymlpath.getPath());
			InputStream is;
			is = resources.openRawResource(R.raw.orb4k_p15e15); //Inout stream for yml file opened
			try{
				byte[] buffer = new byte[is.available()]; 
				is.read(buffer);
				FileOutputStream fos = openFileOutput("orb4k_p15e15.yml", Context.MODE_PRIVATE);
				fos.write(buffer);
				fos.close();
			}
			catch(Exception e){
				windup();
			}
		}
	}

	private Camera getCameraInstance() {

		//Return the instance of camera object after enabling it
		Log.d(TAG,"Obtaining instance of camera");
		Camera camera = null;
		try {
			camera = Camera.open();
		} catch (Exception e) {
			finish();
		}
		return camera;
	}

	PictureCallback mPicture = new PictureCallback() {
		//Operations that happen if a picture is taken
		@Override
		public void onPictureTaken(byte[] data, Camera camera) {
			Log.d(TAG,"picture has been taken");
			pictureFile = getOutputMediaFile();//File path 

			if (pictureFile == null){
				return;
			}

			try {
				FileOutputStream fos = new FileOutputStream(pictureFile);
				fos.write(data);
				fos.close();
			}catch (FileNotFoundException e) {
				windup();
			}catch (IOException e) {
				windup();
			}
			finally{
				count++;
			}
			val = getimage(vocabpath,fpath); //The result is obtained
			Log.d(TAG, "Decision obtained - media player should start now"); 
			if((int)val==-1){
				//Insufficient image data for the image processing code
				Log.d(TAG,"Bad quality - take again");
				playAudio(-2);
			}
			else{
				values[2]=val-10*((int)val/10);val/=10;
				values[1]=((int)val)%10;
				values[0]=(int)(val/10);
				cycleCount++;
				if((int)values[0]==0){
					//A weak result has been achieved - the counting array will be incremented
					choices[(int)values[1]][0]+=1;//Vote up by 1
					choices[(int)values[1]][1]+=values[2];//Weight increased according to native function
					if(choices[(int)values[1]][0]==2){//A majority has been obtained
						playAudio((int)values[1]);//Corresponding audio file played
						resetCycle();
					}
					else if(cycleCount==3){
						//3 different results so the least weighted one will be taken
						int i,j=0;
						float min=1000.0f;
						for(i=0;i<6;i++){
							if(choices[i][1]<min&&(int)choices[i][0]>0){
								min=choices[i][1];
								j=i;
							}
						}
						playAudio(j);
						resetCycle();
					}
					else{
						//User asked to take another picture
						Log.d(TAG,"Doubtful - take another");
						playAudio(-1);
					}
				}
				else if((int)values[0]==1){
					//Strong result - played immediately
					playAudio((int)values[1]);
					resetCycle();
				}
			}
			val=-1.0f;
			flag=true;
		}     
	};

	AutoFocusCallback autoFocusCallback = new AutoFocusCallback() {
		@Override
		public void onAutoFocus(boolean success, Camera camera) {
			Log.i(TAG,"Camera has been focussed"); 
			try{
				Thread.sleep(500);
			}
			catch(Exception e){;}
			mCamera.takePicture(null,null,mPicture); 
		}
	};

	private void resetCycle(){
		cycleCount=0;
		for(int i=0;i<6;i++){
			choices[i][0]=choices[i][1]=0.0f;
		}
	}
	private void playAudio(int n){

		//Plays the audio file corresponding to the result
		switch(n){
		case -2:mplayer=MediaPlayer.create(this,R.raw.bad_note);
		Log.d("MyCameraApp", "Bad note - take picture again");
		break;
		case -1: mplayer=MediaPlayer.create(this,R.raw.take_again);
		Log.d("MyCameraApp", "Take picture again");
		break;
		case 0: mplayer=MediaPlayer.create(this,R.raw.ten);
		Log.d("MyCameraApp", "Answer is ten");
		break;
		case 1: mplayer=MediaPlayer.create(this,R.raw.twenty);
		Log.d("MyCameraApp", "Answer is twenty");
		break;
		case 2: mplayer=MediaPlayer.create(this,R.raw.fifty);
		Log.d("MyCameraApp", "Answer is fifty");
		break;
		case 3: mplayer=MediaPlayer.create(this,R.raw.hundred);
		Log.d("MyCameraApp", "Answer is hundred");
		break;
		case 4: mplayer=MediaPlayer.create(this,R.raw.five_h);
		Log.d("MyCameraApp", "Answer is five hundred");
		break;
		case 5: mplayer=MediaPlayer.create(this,R.raw.thous);
		Log.d("MyCameraApp", "Answer is thousand");
		break;

		}

		if(mplayer!=null)mplayer.start();
		else Log.e(TAG,"Media player could not start");
	}


	private static File getOutputMediaFile(){

		Log.d(TAG,"Picture taken will be stored in directory");
		//Obtains the file path where the taken image will be saved
		File mediaStorageDir = new File(Environment.getExternalStoragePublicDirectory(
				Environment.DIRECTORY_PICTURES), "MyCameraApp");

		if (! mediaStorageDir.exists()){
			if (! mediaStorageDir.mkdirs()){
				Log.e(TAG, "failed to create directory");
				return null;
			}
		}

		// Create a media file name
		String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
		File mediaFile;
		fpath = mediaStorageDir.getPath() + File.separator +
				"IMG_"+ timeStamp +".jpg";
		mediaFile = new File(fpath);

		return mediaFile;
	}

	public void onResume()
	{
		Log.d(TAG,"MainActivity resumes");
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_5, this, mLoaderCallback);
		mSensorManager.registerListener(this, mSensor, SensorManager.SENSOR_DELAY_NORMAL);

	}

	@Override
	protected void onPause() {
		Log.d(TAG,"MainActivity is paused");
		super.onPause();
		mSensorManager.unregisterListener(this);
	}

	public void onSensorChanged(SensorEvent event){

		Log.d(TAG,"Inside onSensorChanged function");
		//Performs certain actions based on the accelerometer
		final float alpha = 0.8f;

		// Isolate the force of gravity with the low-pass filter.
		gravity[0] = alpha * gravity[0] + (1 - alpha) * event.values[0];
		gravity[1] = alpha * gravity[1] + (1 - alpha) * event.values[1];
		gravity[2] = alpha * gravity[2] + (1 - alpha) * event.values[2];

		// Remove the gravity contribution with the high-pass filter.
		lin_acc[0] = event.values[0] - gravity[0];
		lin_acc[1] = event.values[1] - gravity[1];
		lin_acc[2] = event.values[2] - gravity[2];

		double d = Math.abs(lin_acc[0]-oldvals[0])+Math.abs(lin_acc[1]-oldvals[1])+Math.abs(lin_acc[2]-oldvals[2]);

		if(flag==false&&d<LOW_THRESH){
			count++;
			//If the phone is held mostly still for 3-4 seconds, a picture is taken
			if(count==limit){
				mCamera.autoFocus(autoFocusCallback);

			}
		}

		else if(flag==true&&d>HI_THRESH){
			//Has been shaken hard so preview must restart
			mCamera.startPreview();
			count = 0;
			flag=false;

		}
		else{
			//Phone is just being moved around
			count=0;
		}

		for(int i=0;i<3;i++){
			oldvals[i] = lin_acc[i];
		} 


	}

	@Override
	public final void onAccuracyChanged(Sensor sensor, int accuracy) {
		// Do something here if sensor accuracy changes.
	}

	public void windup(){
		if(mplayer!=null)mplayer.release();
		finish();

	}

	private boolean checkCameraHardware() {
		if (this.getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA)){
			// this device has a camera
			return true;
		} else {
			// no camera on this device
			return false;
		}
	}

	OnAudioFocusChangeListener afChangeListener = new OnAudioFocusChangeListener() {
		public void onAudioFocusChange(int focusChange) {
			if (focusChange == AudioManager.AUDIOFOCUS_LOSS_TRANSIENT_CAN_DUCK) {
				mplayer.setVolume(1.0f,1.0f);
				// Lower the volume
			} else if (focusChange == AudioManager.AUDIOFOCUS_GAIN) {
				mplayer.setVolume(1.0f,1.0f);
			}
		}
	};

	public native float getimage(String ymlpath, String fpath);

}
