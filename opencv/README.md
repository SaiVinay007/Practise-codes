
**General Techiniques**:

    morphology:processing digital images ,for processing geometric structures 
    erosion:removing boundary pixels of objects, makes objects thin  =>  erosion = cv2.erode(img,kernel,iterations = 1)(grayscale img)
    dilation:inverse of erosion , increases bject boundary to background,thickens objects  =>  dilation =     cv2.dilate(img,kernel,iterations = 1)  
   
    opening:erosion followed by dilation     =>      opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing:dilation followed by erosion     =>      closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((5,5),np.uint8)
    Morphological Gradient(difference between dilation and erosion of an image,used to find edges) =>  gradient =  cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    Distance transformation: fades images ; process of representing the foreground of a binary image,
                         with the distance from nearest obstacle or background pixel.it has the effect of
                         fading out the edges of the foregrounds and removing small islands of foreground,while 
                         preserving the foreground centers

    Translation: Its a matrix addition to each and every pixel of image (moving every pixel)
    Rotation: Its a matrix multiplication of rotation matrix with each pixel . cost -sint   (rotation matrix) 
                                                                           sint  cost 
                                                                           
     Homography:
     Affine transform: any transform of form ax+b 
     Perspective transform:we give four points(in image) and  give the destination points (a rectangle) finds a corresponding      matix which when apllied on image we get the front view





**Steps in digital image processing**
    
    image acquirisition
    preprocessing : lens correction, artifact removal,gray scaling, lighting adjustment,de-blurring and de-noising.
    segmentation : extract object of intersect,etc
    interpretation : stay between lines of the road .For example, if we've been using something like a tiny edge detector,for an autonomous driving application,we're hoping the interpretation is to stay
                     between the lines, and in our lane. 
    result : chemoteraphy,etc.In a medical imaging example,if we've decided there'sa tumor in a medical image, hopefully,
             there's a result, where there's a course of treatment for that.


**IMAGE PRE-PROCESSING**:

    Image Resampling:
        Downsampling : Our camera could be pre-defined for us and often is, in which case, we could have a 4K video feed at 60 frames per second to find pedestrians in.
                       So in that case, simply for processor load, we might have to re-sample the image to 10 APP and 25 frames per second for example.
        Upsampling   : Similarly, we might need to up sample an image.For instance, we could be dealing with a low cost IR sensor with relatively few pixels,
                       in which case, we need to think about how we might upsample that image to fit what our algorithms are used to.

    Conversion to grayscale
    Contrast enhancement : We take the individual RGB values of each of the pixels in the image and plot a histogram of these values. If the histogram has space,
                           and it typically does out at the edges,we can create a new histogram,which more fully uses the available space,
                           and map our original values into this new histogram.This results in more perceived contrast.
    corrections for illumination
    Lens effects / medium effects
        Lens distortion
        Artefact removal
    Denoising / noise filtering     OpenCV's fastN1MeansDenoisingColored routine to denoise the image.
                    denoised image =      cv2.fastNlMeansDenoisingColored(img,None,h,hColor,templateWindowSize,searchWindowSize)
    Blurring


**Segmentation Applications**:

    Face recognization
    Emotion recognization
    Driver fatigue
        eye detection
    Hand detection
    Gesture recognization
    product inspection
    license plate recognization
    traffic management
    Subsea inspection
    entertainment industry
    Satellite imagery
    security industry
    Automotive
        Brake light detection
        Pedestrian detection
        Lane detection
        Speed limit detection

**Five main categories of Image Segmentation algorithms**:
    
    Thresholding
    Clustering
    Region growing
    Edges,corners
    Templates
    And of course, Neural Network based

**ThresHolding**:
    
    Supervised : trial and error method
    
    Otsu's method(unsupervised) : Otsu's method exhaustively iterates through all possible thresholds values, and calculates the variance for
                                  the pixel levels on either side of the threshold, i.e. the pixels that fall either into the foreground or the background at that threshold.
                                  The goal is to find the threshold value, where the sum of the background and foreground variances is at its minimum.
                                  For this, our cv2.threshold() function is used, but pass an extra flag, cv2.THRESH_OTSU. For threshold value, simply pass zero. Then the algorithm finds the optimal threshold value and returns you as the second output, retVal.
                                  If Otsu thresholding is not used, retVal is same as the threshold value you used.

                                  # Otsu's thresholding
                                    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                                  # Otsu's thresholding after Gaussian filtering
                                    blur = cv2.GaussianBlur(img,(5,5),0)
                                    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    
    Adaptive Local Thresholding :  uses techniques to divide the image into smaller segments, and sets a different threshold for each segment.
                                   Adaptive Method - It decides how thresholding value is calculated.
                                   cv2.ADAPTIVE_THRESH_MEAN_C : threshold value is the mean of neighbourhood area.
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C : threshold value is the weighted sum of neighbourhood values where weights are a gaussian window.
                                   Block Size - It decides the size of neighbourhood area. 




    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)                                      =  Global Thresholding (v = 127)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\cv2.THRESH_BINARY,11,2)     = Adaptive Mean Thresholding                                    
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\cv2.THRESH_BINARY,11,2) = Adaptive Gaussian Thresholding                                       
    to show image in black and white : plt.imshow(img,'gray')
    thresholding is done for grayscale images

**Clustering** : 
        k-means,Fuzzy C-means,Expectation maximization.
        k-means: take initial k points then by finding distance form k clusters,then find centers of each cluster again repeat     the same process untill ,it stops changing much.
        




**Sobelx** : we apply convolution to our grayscale image (kernel be like [[-1 0 1],[-4 0 4],[-1 0 1]]) to find gradients in our image along x direction then we threshold to find most likely edges  
**Sobely** : same as sobelx but kernel will be like [[-1 -4 -1],[0 0 0],[1 4 1]]


                        



**Canny Edge detection** :
    
    Gaussianfilter
    Find intensity gradient of image : at each pixel we find Gx,Gy and sqrt(Gx,Gy) and direction tan-1(Gy/Gx) and round of the directions to 8
    Non-maximum suppression : Thinning of edges by considering only the pixels(comparing with pixels only in the kernal) which are local maxima along the gradient direction
    Double thresholding: for removing noise.
                          If the pixel gradient is higher than the upper threshold, it is marked as a strong edge pixel.
                          If the pixel gradient is between the two thresholds, it is marked as a weak edge pixel.
                          If the pixel gradient value is below the lower threshold, it will be suppressed.


    Hysteresis :For in weak edge pixels,go along edge (retained ones) if it is conneted to these pixels then retain, else discard.




**1.Object Detection**:

    #######################
    Corner Detection : 
                        corner means gradient of intensity of pixels is more in more than one direction(different along two perpendicular directions),genrally invarient to translation,rotation,intensity.
    we check for corners for each pixel.If pixel is in a flat region, nearby paches look the same.If image is on edge,one side of edge and other side of edge look different.
    For a corner all patches nearby look  different.So we take a patch(a kernel) and find the sum of squared differences(SSD).If it is less it implies the patches are more Similarl
    and to find a corner , we take differences along two perpendicular directions and ensure its above some threshold value.
    E(u,v)=∑x∑yw(x,y)(I(x+u,y+v)−I(x,y))2 ,where u,v are any two perpendicular directions.From the Spectral theorem in linear algebra, we know that any symmetric matrix can be represented as follows M=Q.D.QT . M=∑x,yw(x,y)[[IxIx,IxIy][IxIy,IyIy]
    Finally,
        E(u,v)=[u′v′]D[u′v′]T=λ1(u′)2+λ2(v′)2 ; λ1 and λ2 are eigen values of M.
    For gradient to be high along both directions,λ1 and λ2 must be high.
    Harris corner detector - R=det(M)−k (trace(M))2 where det(M)=λ1λ2 ,trace(M)=λ1+λ2 .  Shi-Tomasi is another corner detection algorithm.

    ########################
    SIFT :
            SIFT feature descriptor is invariant to uniform scaling, orientation, illumination changes, and partially invariant to affine distortion.



    ########################
    HOG : 
            we make image of sixe 64x128 .then make grid cells of 8x8 each.we calculate histogram of gradients for each cell and decribe it as a feature vector of size 9x1.
            we normalize the a block of histograms(2x2) by L1-norm or L2-norm, getting a feature vector of size 36x1.
            
            Compute a Histogram of Oriented Gradients (HOG) by
            global image normalisation
            computing the gradient image in x and y
            computing gradient histograms
            normalising across blocks
            flattening into a feature vector


    #######################
    Deformable parts models :  



**2.Action Recognition**:
    

    Optical flow:
                    Optical flow works on several assumptions: The pixel intensities of an object do not change between consecutive frames.
                                                               Neighbouring pixels have similar motion. 

                    I(x,y,t) = I(x+dx, y+dy, t+dt) (as intensity remains constant).(optical flow equation).

                    I_x * u + I_y * v + I_t = 0     (by derivating the RHS wrt time) where I_x = d/dx(I)(partial) similarly I_y ;u = dx/dt

                     In it, we can find I_x and I_y, they are image gradients. Similarly I_t is the gradient along time. But (u,v) is unknown . We cannot solve this one equation with two unknown variables.
                     So several methods are provided to solve this problem and one of them is Lucas-Kanade and other is  Horn-Schunck.

                     We have seen an assumption before, that all the neighbouring pixels will have similar motion. Lucas-Kanade method takes a 3x3 patch around the point.
                     So all the 9 points have the same motion. We can find (f_x, f_y, f_t) for these 9 points. So now our problem becomes solving 9 equations with two unknown variables which is over-determined. A better solution is obtained with least square fit method

                    its like Av = b ; A is matrix of Ix,Iy (9x2 if we consider a patch of 3x3) and v is a matrix 0f x,y direction velocities(u,v) and b is matrix of It.
                    we find u,v by using least square fit(linear regression).
    
    ############################
    Background Substraction methods :

    ############################
    Space time interest points : 

    ############################
    Poselets :




**3.Scene Recognition** :
    
    ##################
    Gist :
    ##################
    Spatial Pyramid : 
    ##################
    Indoor scenes ,Mid level patches,Scene DPM :



**4.Saliency Estimation** : 
    
    Itti-Koch Saliency model :
    Learning to Predict :
    Distinct patches,Category Independent :



**5.Sematic Segmentation** :



**6.HarrCascade** : 



**7.Histogram Backprojection** :
                               
    It is used for segmentation and finding object of our interest in image.It gives us a single channeled image with our required being white and remaining black.
    We create a histogram of an image containing our object of interest.And a color histogram is preferred over grayscale histogram, because color of the object is a better way to define the object than its grayscale intensity.
    We backProject the histogram on the target image which gives returns the probability image.
                                



**8.Kalman Filter** :



**9.CamShift** :
                
    Continuously Adaptive Meanshift
    It applies meanshift first. Once meanshift converges, it updates the size of the window as, s=2×√‾‾‾‾M00/256. It also               calculates the orientation of best fitting ellipse to it.
    Again it applies the meanshift with new scaled search window and previous window location.
    The process is continued until required accuracy is met.



**10.Template Matching** :
                            
    It is technique for finding areas of the image that are similar to the patch(Template).We should set high threshold to get accurately get same features in image.
    For example of if we are applying face recognition and we want to detect the eyes of a person, we can provide a random image of an eye as the template and search the source (the face of a person).
    In this case, since “eyes” show a large amount of variations from person to person, even if we set the threshold as 50%(0.5), the eye will be detected.

    The template image simply slides over the input image (as in 2D convolution)
    The template and patch of input image under the template image are compared.
    The result obtained is compared with the threshold.
    If the result is greater than threshold, the portion will be marked as detected.

    In the function cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED) the first parameter is the mainimage, second parameter is the template to be matched and third parameter is the method used for matching.

    Multi-scale template matching:



**11.Contours** : 



**12.Hough Lines** : 
                    
    Hough Transform is a popular technique to detect any shape, if you can represent that shape in mathematical form. It can detect the shape even if it is broken or distorted a little bit.
    We try to find the values of (r,theta) instead of (b,m) as vertical lines gives uncertain values of m.
    In its original formulation , each edge point votes for all possible lines passing through it, and lines corresponding to high accumulator or bin values are examined for potential line fits.

    Size of array depends on the accuracy you need. Suppose you want the accuracy of angles to be 1 degree, you need 180 columns. For \rho, the maximum distance possible is the diagonal length of the image.
    So taking one pixel accuracy, number of rows can be diagonal length of the image.

    We for every point on edge we increment theta and find corresponding rho .If the same value pair repeats, then we increse its votes(preference) and at final we take the pair having maximum votes maximum  



**13.Foreground Extraction** : 



**14.Region of growth**:



**15.Mean Shift Tracking** :
                            
    It is also known as Kernel-Based tracking.  It  is  an  iterative  positioning  method  built  on  the augmentation of a parallel  measure  (Bhattacharyya coefficient). 
    when the target moves so fast that the target area in the two neighboring frame will not overlap, tracking object often converges to a wrong object.
    For this problem, traditional Mean shift algorithm easily failed to track fast moving object, some solutions had proposed like combining Kalman filter or Particle filter with Mean shift algorithm.



**16.Dense motion estimation** :



**17.OCR** :



**18.Histogram Equalization**: 
                            
    histogram equalization can be used to enhance contrast.
    First we have to calculate the PMF (probability mass function) of all the pixels in this image.
    Our next step involves calculation of CDF (cumulative distributive function).
    To make it clearer, from the image above, you can see that the pixels seem clustered around the middle of the available range of intensities. What Histogram Equalization does is to stretch out this range.
                            


**19.Viola-Jones** :



**20.SURF** :
