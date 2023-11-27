This dataset contains touch input data collected from 89 children and 30 adults using a smartphone and a tablet.

Each folder represents one participant. 

Each folder contains four XML files (corresponding to four tasks, one file per task type): tap.xml, doubletap.xml, singletouch-draganddrop.xml, and multitouch-draganddrop.xml. Please see the article Vatavu et al. (2015), Int. J. Human-Comp. Studies, for a description of these tasks and the experimental setup.

Each XML file contains data collected during several trials (the number of trials is not the same for all the participants; see the article for information regarding the completion rate of the experiment). Each task is represented as a series of touch strokes with X and Y coordinates and timestamps T. 

--------!!!!!!!!!!!!!!!!-----------------
IMPORTANT: Trials in each file are not listed in the chronological order as performed, e.g., the first trial from a file is not necessarily the first one performed by that participant. The chronological order of the trials was unfortunately lost. Therefore, this dataset cannot be used to analyze, for instance, the effect of practice on participants' touch accuracy. 
--------!!!!!!!!!!!!!!!!-----------------

The companion source code (ComputeTouchMeasurements) reads this dataset and computes the measurements reported in the article, such as Tap-Time, Tap-Accuracy, DoubleTap-Time, DoubleTap-Accuracy, etc.
