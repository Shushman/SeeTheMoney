#include <jni.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/internal.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <android/log.h>
#include <sys/types.h>

using namespace std;
using namespace cv;


int getResult(Mat&);//Obtains the probable result, given a candidate image - 10 is index 0, 20 is 1, and so on
void getCannyParams(Mat&,int&,int&);//Obtains Canny edge parameters based on grayscale values
Mat getSegmented(Mat);//Obtains a segmented version of the image around the note
Mat c_test;

Mat submats[6];//6 matrices for data of each clusters
Mat labmats[6];//6 label matrices containing the label of each cluster
CvKNearest knns[6];//For obtaining the nearest neighbour in each cluster
TermCriteria criteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 0.1);
float values[3] = {0};


Mat& cluster_test(Mat& test_mat,int clusters=50)
{

	if(test_mat.rows > clusters)
	{
		BOWKMeansTrainer bowtrainer(clusters,criteria,1,KMEANS_PP_CENTERS); //num clusters
		bowtrainer.add(test_mat);
		c_test = bowtrainer.cluster();
	}
	else
	{
		c_test=test_mat.clone();
	}
	__android_log_write(ANDROID_LOG_INFO, "JNI", "Query image clustered");
	return c_test;
}


int getResult(Mat& test_image){

	double res[6] = {0};//To store the total weighted votes based on distance
	double mins[6];//To store the votes for each descriptor
   	Mat descriptors;
	__android_log_write(ANDROID_LOG_INFO, "JNI", "Beginning feature detection and extraction");

	OrbFeatureDetector detector(500, 1.2f,8,15,0,2,0,15);
	FREAK extractor(true,true,15);

	Mat nearest=cvCreateMat(1, 1, CV_32FC1);//To store the label of the nearest cluster (trivial parameter)
 	Mat neighbor_dist=cvCreateMat(1,1, CV_32FC1);//To store the nearest neighbour distance
  	vector<KeyPoint> keypoints;//For the key-points detected

  	detector.detect(test_image,keypoints);//Detection of key-points
  	extractor.compute(test_image, keypoints, descriptors);//Extraction of descriptors

  	Mat des1,des2;
  	int i,j,k,l,c,temp;
	double dist,min;

	float response;
  	descriptors.convertTo(des1,CV_32FC1);

  	//The module below votes for the possible results depending on distances from each cluster
	if(des1.rows>120){
		c = des1.rows/5;
		des2 = cluster_test(des1,c);
  		for(i=0;i<des2.rows;i++){
			for(j=0;j<6;j++){
				response = knns[j].find_nearest(des2.row(i),1,0,0,&nearest,&neighbor_dist);
				mins[j]=(neighbor_dist.at<float>(0,0));
			}
			//The minimum distance is obtained and the other distances' ratio to the minimum taken
			min=mins[0];
			for(j=1;j<6;j++){
				if(mins[j]<min)
					min=mins[j];
			}
			for(j=0;j<6;j++){
				res[j]+=mins[j]/min;
			}
		}
  		int ind[6] = {0,1,2,3,4,5};
  		//The index with the minimum votes is the likely solution
		for(i=0;i<5;i++){
			for(j=i+1;j<6;j++){
				if(res[i]>res[j]){
					response=res[i];res[i]=res[j];res[j]=response;
					temp=ind[i];ind[i]=ind[j];ind[j]=temp;
				}
			}
		}
		values[0]=ind[0]; values[1]=res[0]/c;
		if(res[1]-res[0]>c/20.0){
			return 1;
		}
		else{
			return 0;
		}
	}
	else
  	{
		values[0]=values[1]=-1;
  		return -1; //Fewer than 60 descriptors will require the user to take another image
  	}
}


void getCannyParams(Mat& src, int& low, int& high){

	__android_log_write(ANDROID_LOG_INFO, "JNI", "Obtaining parameters for Canny edge detection");
	int i,j,tot=src.rows*src.cols,pos,sum=0;
	int bins[256] = {0};

	for(i=0;i<src.rows;i++){
		for(j=0;j<src.cols;j++){
			pos = (int)(src.at<uchar>(i,j));
			bins[pos]++;
		}
	}
	for(i=0;i<256;i++){
		sum+=bins[i];
		if(sum>tot/2)//Median pixel value
			break;
	}
	low=(5*i)/12;
	high = 2*low;
}


Mat getSegmented(Mat src){

	__android_log_write(ANDROID_LOG_INFO, "JNI", "Segmenting the image");
	Mat copy = src.clone();
	Mat res, dst;
	int threshold=100,lt=50,ht=230,step=10;
	GaussianBlur(src,src,Size(3,3),0,0);//Image smoothed to remove small distortions

	int minx,miny,maxx,maxy;
	getCannyParams(src,lt,ht);//Canny edge parameters obtained
	vector<Vec2f> lines;
	vector<Point2f> intersections;
	Canny(src,dst,lt,ht,3);
	int x1,y1,l=10000,i,j,target;
	__android_log_write(ANDROID_LOG_INFO, "JNI", "Detecting lines with Hough Transform");

	//Houghlines is run repeatedly till the number of lines is reasonable
	HoughLines(dst, lines, 1, CV_PI/180, threshold, 0, 0);
	l = lines.size();
	target = (l/10>500)?l/10:500;

	while(l>target){
		threshold+=step;
		HoughLines(dst, lines, 1, CV_PI/180, threshold, 0, 0 );
		l = lines.size();
	}

	//Points of intersection of perpendicular lines are considered
	for(i = 0; i< l; i++ ){
		float rho1 = lines[i][0], theta1 = lines[i][1];
		for(j=0;j<l;j++){
			if(j==i) continue;//Same lines are not considered
			float rho2,theta2;
			rho2 = lines[j][0]; theta2 = lines[j][1];
			float diff = abs(theta1-theta2);
			if(abs(diff-1.57)>0.05)//Lines should be almost perpendicular
				continue;
			y1 = (int)abs((rho1*cos(theta2)-rho2*cos(theta1))/sin(theta1-theta2));
			x1 = (int)abs((rho1*sin(theta2)-rho2*sin(theta1))/sin(theta2-theta1));

			if(x1>0&&x1<src.cols&&y1>0&&y1<src.rows){
				intersections.push_back(Point2f(x1,y1));
			}
		}
	}

	l = intersections.size();
	if(l==0){
		//No intersections could be accurately computed, so whole image taken
		__android_log_write(ANDROID_LOG_INFO, "JNI", "No intersections in image");
		res = copy(Range(60,src.rows-60),Range(80,src.cols-80));
		return res;
	}

	//The maximum bounding box from intersection points is considered
	Point2f p = intersections[0];
	minx=maxx=p.x;miny=maxy=p.y;
	for(i=1;i<l;i++){
		Point2f p =intersections[i];
		if(p.x<minx) minx=p.x;
		else if(p.x>maxx) maxx=p.x;
		if(p.y<miny) miny=p.y;
		else if(p.y>maxy) maxy=p.y;
	}
	//In case the bounding box is too small
	if(maxy-miny<400||maxx-minx<400){
		res = copy(Range(60,src.rows-60),Range(80,src.cols-80));
		return res;
	}

	res = copy(Range(miny,maxy),Range(minx,maxx));
	return res;

}

extern "C" {

JNIEXPORT float JNICALL Java_com_example_testocv3_MainActivity_getimage(JNIEnv* env, jobject  obj, jstring ymlpath, jstring fname);

JNIEXPORT float JNICALL Java_com_example_testocv3_MainActivity_getimage(JNIEnv* env, jobject obj, jstring ymlpath, jstring fname)
{
	//The filename for the image as stored in the phone, is read here
	const char *str = env->GetStringUTFChars(fname,0);
	Mat img = imread(str,0);
	env->ReleaseStringUTFChars(fname,str);

	//The vocabulary file in memory is read here
	str = env->GetStringUTFChars(ymlpath,0);
	FileStorage fs(str,FileStorage::READ);
	env->ReleaseStringUTFChars(ymlpath,str);
	__android_log_write(ANDROID_LOG_INFO, "JNI", "Vocabulary file has been loaded");

	int i,j;
	Mat train_mat;
	Mat label_mat;
	fs["train_mat"] >> train_mat;//The training data matrix
	fs["label_mat"] >> label_mat;//The label data matrix

	//The training and label matrices for each cluster are populated
	for(i=0;i<train_mat.rows;i++){
		j=(int)(label_mat.at<float>(i,0));
		submats[j].push_back(train_mat.row(i));
		labmats[j].push_back(label_mat.row(i));
	}

	//The CvKNearest objects are trained correspondingly
	for(i=0;i<6;i++){
		knns[i].train(submats[i],labmats[i]);
	}

	if(!img.data){
		__android_log_write(ANDROID_LOG_INFO, "JNI", "Query image did not load properly");
	}
	Mat img2,red;
	resize(img,img2,Size(800,600));
	Mat img3=getSegmented(img2);
	double fact = img3.cols/450.0;
	resize(img3,red,Size(450,(int)(img3.rows/fact)));
	int val = getResult(red);
	float res;
	if(val==-1)
		return -1.0f;
	else{
		res=100*val+10*values[0]+values[1];
	}
	__android_log_write(ANDROID_LOG_INFO, "JNI", "Exiting native code");
	return res;
}
}




