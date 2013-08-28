SeeTheMoney
===========

The Android Application for the app See The Money which recognizes Indian Currency notes from 10 to 1000. The uploaded package is the full Eclipse Project that went into creating the app. The .apk file may be found in the bin folder which
can then be directly ported to an Android phone.

Ver - 2.3 (Gingerbread) upwards
N.B - This is a testing version of the application, and work is still left to be done to make it usable on
a large scale. 


APP DESCRIPTION

The app has a custom touch-based interface for use by the visually impaired. Once the app is started 
(presumably using Screen Reader or some related application), the back camera of the phone automatically starts. 
The note should be held in the view of the camera (the visually impaired can get used to this after practicing 
a few times) and the phone should be held fairly steady for a few seconds, after which the photo will be taken.
Then 3 situations may arise:

Case 1 (Easy Result) - The photo quality is good and the note is easily identified. The app reports the result after which the user
can shake the phone 2-3 times and resume camera feed.

Case 2 (Reasonable Doubt) - Due to lower quality of the note/image or occlusion, the app detects multiple possibilities
for the note. The message "Please take the photo again" will be played after which the user should take a different
side of the SAME note and take another photo. This will have to be done a maximum of three times if needed.

Case 3 (No Chance) - Due to the note not being there in the image or an extremely low quality of the image, the app cannot
identify any possibility for the note. The message "Bad Image! Please adjust the note" will be played.

To close the app, the user must touch the screen. Therefore it is recommended to hold the side of the phone
while testing the app.


CODE DESCRIPTION

1. Front End (In the src folder) - This is the camera interface which allows the user to take photos automatically
based on the user's movements. This is coded in Java using the Android Development Toolkit methods.

2. Back End (In the jni folder) - This is the computer vision and analysis part. It is coded in C++ and uses the 
OpenCV4Android package via the Java Native Interface. 

3. Learned Model - The training was done using labelled K-means clustering with around 4000 clusters. The information 
is stored as a mtrix in the .yaml file which is then imported at the start of runtime via OpenCV. The checking is done
by a weighted K-nearest neighbour method considering each label individually. 
