# VisualDiary

In this post I will describe a fun lifelogging application I built over the fall semester. The application would summarize a user’s geo-tagged video feed. With the relapse of wearable cameras, I thought this was an interesting project to work on. Personally, I think it would be essential to do so if the camera were to continuously record. Why? At 30 fps a camera that is always recording would capture roughly 2.5 million frames. But very few of these would correspond to different activities. Sifting through these manually would be almost impossible for a human.

My motivation to use deep learning to solve the video summary problem came from a report I read after completing the cs231n course. To make the summary more representative of the entire journey, I decided to overlay GPS data on the video summary.

Video Summary

To generate a summary of a video you would have to identify frames that are most different from one another. The most naive way to do this would be to just subtract every frame of the video from one another and pick the frames which give the highest difference. But this would mean that consecutive frames could give a very high difference even if there is only a marginal change in the position of the same objects. Not a very good summary. How to get around this?

In order to build a summary this difference should not be high if the consecutive frames contain the same objects only displaced or warped in the image. Put another way, the frame must be transformed to a set of features that will be very similar for the same object regardless of its position or orientation in the image. This can then be reduced to an image classification problem. CNN’s are a natural fit for this as they are robust to such change in the position of an object in the image. Using a network that has been pre trained with a large and diverse dataset, we should be able to extract features for new images in the video to be summarized. Hence, instead of directly taking a difference of raw pixels, the features extracted from the CNN are used for each image. Specifically, an image would be fed to the network and the output of the last layer of the network is extracted and used to find the difference.

This video shows a comparison of using raw pixels versus extracted features for the task of video summarization. 

The number of such salient images to be detected is bounded to a fixed number. Hence, if this number was 5, frames which correspond to the highest 5 differences are chosen as salient images in the video.

Why use GPS?

Using change points alone would over represent video segments with more objects and possibly miss out on activities like working on a desk. In order to make the summary representative of more activities, I decided to use GPS data to demarcate periods of motion and rest in the video. This demarcation was then used to enforce a fixed number of change points in each period of rest and motion. 

I plotted the latitude and longitude against time and fixed a threshold for both. If the variation was below this threshold for both, the user was at rest. 
