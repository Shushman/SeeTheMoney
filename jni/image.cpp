#include <jni.h>
#include <cstdio>
#include <iostream>
#include<math.h>
#include<ml.h>
#include <cstring>
#include<ctime>
#include <android/log.h>
#include <sys/types.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/opencv.hpp"

const int WIDTH = 480;
const int HEIGHT = 320;
const int TRAIN_IMAGE = 10;
const int VOCAB_SIZE = 10000;

using namespace cv;
using namespace std;

SiftFeatureDetector detector(500);
BOWImgDescriptorExtractor bowDE(new SiftDescriptorExtractor(),new FlannBasedMatcher());
Mat vocabulary, inverted_index[VOCAB_SIZE],labels;
Mat keypoints_imgs;
string keypath;
float *dot_pro;int *indices;
int geo_score[20]={0};
int correct=0;
int decision[6]={0};
int testval = 5;


int max_decision()
{
	//Returns the most likely decision. Each integer maps to a particular currency note, or an unsure result
	int max_index=-1;
	int max=0;
	for(int i=0;i<6;i++)
	{
			if(max<decision[i])
			{
					max_index=i;
					max=decision[i];
			}
	}
	if(max<60) //Uncertain result
		return -2;
	return max_index;
}

Mat grabcut_seg(Mat input)
{
		//Segmentation using two iterations of GrabCut
	    Mat inp2;
        resize(input,inp2,Size(640,(int)(input.rows*(640.0/input.cols))));
        Mat img;
        resize(input,img,Size(90,(int)(input.rows*(90.0/input.cols))));

        int thresh=(int)(0.9*img.rows*img.cols);
        int count=0;
        Rect rect=Rect(5,5,img.cols-10,img.rows-10);
        Mat  bgdModel, fgdModel;
        Mat mask;

        //First iteration to obtain initial mask
        grabCut(img, mask,rect,bgdModel,fgdModel,1,GC_INIT_WITH_RECT );

        for(int i=0;i<mask.rows;i++)
        {
                for(int j=0;j<mask.cols;j++)
                {
                        if(mask.at<uchar>(i,j)==0||mask.at<uchar>(i,j)==2)
                        {
                                mask.at<uchar>(i,j)=2;
                                count++;
                        }
                }
        }

        if(count>0 && count<thresh) //Within the specified thresholds
        {
                grabCut(img, mask,rect,bgdModel,fgdModel,1,GC_INIT_WITH_MASK );
        }

        else
        {
                for(int i=0;i<mask.rows;i++)
                {
                        for(int j=0;j<mask.cols;j++)
                        {
                                if(i>0.2*mask.rows && i<0.8*mask.rows)
                                {
                                        if(j>0.2*mask.cols && j<0.8*mask.cols)
                                        {
                                                mask.at<uchar>(i,j)=3;
                                        }

                                }
                        }
                }
        }
        cv::compare(mask,cv::GC_PR_FGD,mask,cv::CMP_EQ);

        Mat mask2;
        resize(mask,mask2,Size(inp2.cols,inp2.rows),0,0,CV_INTER_LINEAR);
        return mask2;
}


int keypoints_in_mask(Mat mask,int i,int j)
{
        for(int k=i-1;k<i+1;k++)
        {
            for(int l=j-1;l<j+1;l++)
            {
                if(mask.at<uchar>(k,l)<10)
                {
                    return 0;
                }
            }
        }
    return 1;
}

void remove_keypoints(vector<KeyPoint> &keypoints,Mat mask)
{
	//Removes keypoints not in the mask
    for(int i=0;i<keypoints.size();)
    {
    	int x=keypoints[i].pt.x;
    	int y=keypoints[i].pt.y;

    	if(!keypoints_in_mask(mask,y,x))
    	{
    		keypoints.erase(keypoints.begin()+i);
    	}
    	else
    	{
    		i++;
    	}
   }
}


int arg_sort()
{
	//Ranks the retrieved images based on the similarity with the given image
    indices=(int *) malloc(sizeof(int)*labels.rows);
    for(int i=0;i<labels.rows;i++)
    {
        indices[i]=i;
    }

    int tmp_index; float tmp;

    for(int i=0;i<labels.rows;i++)
    {
        for(int j=i+1;j<labels.rows;j++)
        {
            if(dot_pro[i]<dot_pro[j])
            {
                tmp=dot_pro[i];
                dot_pro[i]=dot_pro[j];
                dot_pro[j]=tmp;

                tmp_index=indices[i];
                indices[i]=indices[j];
                indices[j]=tmp_index;
            }
        }
    }
    for(int i=0;i<10;i++)
    {
    	__android_log_print(ANDROID_LOG_INFO, "JNI","\n%d : %d :%f",i,indices[i],dot_pro[i]);
    }
   return indices[0];

}


int get_top_geoscore()
{
	//Computes the highest score for geometric verification
    int max=0, max_index=0;
    for(int i=0;i<20;i++)
    {
        if(max<geo_score[i])
        {
            max=geo_score[i];
            max_index=i;
        }
    }
    geo_score[max_index]=-10;
    __android_log_print(ANDROID_LOG_INFO, "JNI","\nMax : %d ,%d\n",max,indices[max_index]);fp_k    //printf("\n\nresponse : %d\tcorrect:%d\n\n",response,correct);ey
    decision[labels.at<uchar>(indices[max_index],0)]= decision[labels.at<uchar>(indices[max_index],0)]+max;
    return indices[max_index];
}

void re_rank_geo(vector<KeyPoint> keypoints, vector<vector<int> > pointIdxsOfClusters)
{
	//Performs spatial re-ranking based on geometric verification
    for(int i=0;i<20;i++)
    {
        geo_score[i]=0;
        Mat point1, point2;
        stringstream ss;
        //Reads keypoint locations from folder
        ss<<keypath<<"/"<<indices[i]<<".txt";
        FILE *fp_key;
        fp_key=fopen(ss.str().c_str(),"r");
        int vocab_id=-1,x=-1,y=-1;

        for(int j=0;j<pointIdxsOfClusters.size();j++)
        {
            for(int k=0;k<pointIdxsOfClusters[j].size() && !feof(fp_key);k++)
            {
                Mat temp=Mat(1,2,CV_32F);
                temp.at<float>(0,0)=keypoints[pointIdxsOfClusters[j][k]].pt.x;
                temp.at<float>(0,1)=keypoints[pointIdxsOfClusters[j][k]].pt.y;

                while(!feof(fp_key))
                {
                    fscanf("%d%d%d",&vocab_id,&x,&y);
                    if(vocab_id>j)
                    {

                        break;fp_k    //printf("\n\nresponse : %d\tcorrect:%d\n\n",response,correct);ey
                    }
                    if(vocab_id==j)
                    {
                        Mat temp2=Mat(1,2,CV_32F);
                        temp2.at<float>(0,0)=x;
                        temp2.at<float>(0,1)=y;
                        point1.push_back(temp);
                        point2.push_back(temp2);
                        break;
                    }
                }
            }

        }
	fclose(fp_key);

    if(point1.rows>0)
    {

        Mat out;
        Mat fundamentalMat=findFundamentalMat(point1,point2,FM_RANSAC,3.0,0.99, out );
        for(int j=0;j<out.rows;j++)
        {
            geo_score[i]=geo_score[i]+out.at<uchar>(j,0);
        }
    }
    }

}


int sort_by_dot_product(Mat descriptors)
{
	//Sets up the ranking of similar images by computing the distances based on dot product
    dot_pro=(float *) malloc(sizeof(float)*labels.rows);

    for(int i=0;i<labels.rows;i++)
    {
        dot_pro[i]=0;
    }

    for(int i=0;i<VOCAB_SIZE;i++)
    {
        for(int j=0;j<inverted_index[i].rows;j++)
        {
            dot_pro[ (int)inverted_index[i].at<float>(j,0) ] = dot_pro[ (int)inverted_index[i].at<float>(j,0) ] + inverted_index[i].at<float>(j,1)*descriptors.at<float>(0,i);
        }
    }

    return(arg_sort());


}

int test(Mat img2)
{
	//Obtains the result for a particular image.
    Mat img;
    cvtColor(img2,img,CV_BGR2GRAY);

    __android_log_write(ANDROID_LOG_INFO, "JNI", "Before grabcut");
    Mat mask=grabcut_seg(img2);
    __android_log_write(ANDROID_LOG_INFO, "JNI", "After grabcut");

    if(img.cols>640)
    {
        resize(img,img,Size(640,img.rows*(640.0/img.cols)));
    }
    __android_log_write(ANDROID_LOG_INFO, "JNI", "Before SIFT");
    vector<KeyPoint> keypoints;
    detector.detect(img,keypoints);
    __android_log_write(ANDROID_LOG_INFO, "JNI", "After SIFT detection");

    remove_keypoints(keypoints,mask);
    __android_log_print(ANDROID_LOG_INFO, "JNI", "%d keypoints after masking",keypoints.size());

    if(keypoints.size()<20)
    {
        return -1;
    }

    Mat descriptors;
    vector<vector<int> > pointIdxsOfClusters;
    bowDE.compute(img,keypoints,descriptors,&pointIdxsOfClusters);
    __android_log_print(ANDROID_LOG_INFO, "JNI", "After computing descriptors");
    int dot_result=sort_by_dot_product(descriptors);
    __android_log_print(ANDROID_LOG_INFO, "JNI", "After computing dot product");
    re_rank_geo(keypoints,pointIdxsOfClusters);
    __android_log_print(ANDROID_LOG_INFO, "JNI", "After spatial re-ranking");

    int res=get_top_geoscore();
    int result=labels.at<uchar>(res,0);
    int result1=result;
    for(int i=0;i<10;i++)
    {
        res=get_top_geoscore();
        result=labels.at<uchar>(res,0);
    }

    __android_log_write(ANDROID_LOG_INFO, "JNI", "After geometric verification");

    int response=max_decision();
    if(response==-1)
    {
       return -2;
    }
        for(int i=0;i<6;i++)
        {
                decision[i]=0;
        }
     __android_log_write(ANDROID_LOG_INFO, "JNI", "After decision");
    return response;


}

extern "C" {

JNIEXPORT void JNICALL Java_com_example_seethemoney_MainActivity_loaddata(JNIEnv* env, jobject  obj, jstring ymlpath, jstring indexpath, jstring labelpath, jstring keyptspath);

JNIEXPORT void JNICALL Java_com_example_seethemoney_MainActivity_loaddata(JNIEnv* env, jobject  obj, jstring ymlpath, jstring indexpath, jstring labelpath, jstring keyptspath)
{

	//The vocabulary file in memory is read here
	const char *strvocab = env->GetStringUTFChars(ymlpath,0);
	FileStorage fs_vocab(strvocab, FileStorage::READ );
	env->ReleaseStringUTFChars(ymlpath,strvocab);
	__android_log_write(ANDROID_LOG_INFO, "JNI", "Vocabulary file has been loaded");

	const char *strindex = env->GetStringUTFChars(indexpath,0);
	FileStorage fs_inv(strindex, FileStorage::READ );
	env->ReleaseStringUTFChars(indexpath,strindex);
	__android_log_write(ANDROID_LOG_INFO, "JNI", "Index file has been loaded");

	const char *strlabels = env->GetStringUTFChars(labelpath,0);
	FileStorage fs_labels(strlabels, FileStorage::READ );
	env->ReleaseStringUTFChars(labelpath,strlabels);
	__android_log_write(ANDROID_LOG_INFO, "JNI", "Labels file has been loaded");

	const char *strkeypoints = env->GetStringUTFChars(keyptspath,0);
	keypath=string(strkeypoints);
	env->ReleaseStringUTFChars(keyptspath,strkeypoints);
	__android_log_write(ANDROID_LOG_INFO, "JNI", "Keypoints file has been accessed");

	Mat train_mat;
	Mat label_mat;
	fs_vocab["vocabulary"] >> vocabulary;//The training data matrix
	vocabulary.convertTo(vocabulary,CV_32F);
	fs_labels["labels"] >> labels;//The label data matrix


	for(int i=0;i<VOCAB_SIZE;i++)
	    {
	        char n[50];
	        sprintf(n,"inverted_index_%d",i);
	        fs_inv[n] >> inverted_index[i];
	    }

	fs_labels.release();
	fs_inv.release();
	fs_vocab.release();

	bowDE.setVocabulary(vocabulary);
	__android_log_write(ANDROID_LOG_INFO, "JNI", "Loading of yml files done");

	testval = 10;
}
}

extern "C" {

JNIEXPORT int JNICALL Java_com_example_seethemoney_MainActivity_getimage(JNIEnv* env, jobject  obj, jstring fname);

JNIEXPORT int JNICALL Java_com_example_seethemoney_MainActivity_getimage(JNIEnv* env, jobject  obj, jstring fname)
{
	//__android_log_print(ANDROID_LOG_INFO, "JNI", "Test should be 10 and is %d",testval);
	//The filename for the image as stored in the phone, is read here
	const char *str = env->GetStringUTFChars(fname,0);
	Mat img = imread(str,1);
	env->ReleaseStringUTFChars(fname,str);

	if(!img.data){
		__android_log_write(ANDROID_LOG_INFO, "JNI", "Query image did not load properly");
	}
	int val = test(img);
	__android_log_write(ANDROID_LOG_INFO, "JNI", "Exiting native code");
	return val;
}
}



