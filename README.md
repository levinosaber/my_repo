AD-Census cost based algorithm

Inspired by the method presented in article 
Mei, X., Sun, X., Zhou, M., Jiao, S., Wang, H., & Zhang, X. (2011, November). On building an
accurate stereo matching system on graphics hardware. In 2011 IEEE International Conference
on Computer Vision Workshops (ICCV Workshops) (pp. 467-474). IEEE., 

we decided to implement the stereo matching method based on the AD-Census cost function. 
This algorithm is presented in algo005/stereomatch.py. 
The AD cost function is easy to implement, but it is sensitive to luminance differences. On the other hand, the Census transformation does not require color consistency between neighboring pixels. 
Therefore, it is more robust to radiance variations. The AD-Census combines the advantages of both approaches by merging them. 
The Census algorithm does not perform well with repetitive textures, whereas the AD algorithm, based on individual pixels, can to some extent mitigate the difficulties in processing repetitive textures encountered by the Census algorithm.

Method of use
generate result disparity image:(file path in windows format)
`python .\stereomatch.py im2.png im6.png jieguo2222.png`
replace im2.png by other left image, and im6.png by other right image, replace jieguo2222.png by your result file name

evaluate the result:
`python .\evaldisp.py .\disp2.png .\occl.png .\jieguo2222.png`
disp2.png is the real true disparity image, and occl.png is the image of occultation(both shoulbe be preprared for evaluation)

