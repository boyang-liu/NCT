# Questions (and Answers) Meeting 11-16-2021

## Nico
* ### Date of presentation
Mail

* ### Sample data coordinates (last meeting)
Mail

* ### What is the problem with RANSAC for our usecase?
Multiple model runs could be slow

* ### Opinion on RANSAC models (Essential Matrix, Homography, problems concerning nonlinearity)
Worth trying

* ### Ideas about how we can use Kalman Filters (compared to image stabilization paper)
Maybe there are existing methods

## Flo
* ### Shall we try sth on our own?
Worth trying

## Boyang
* ### How does the Flownet2.0 work for outlier rejection?
Every pixel's rgb and intensity represent the direction and speed about movement from current image to the next.Using these color information can we do the rejection.
* ### In our dataset there are files only in the format ".jpg" and ".exr".How can I implement it with the video format as the input?
Flownet gives the method to compare 2 images. 
