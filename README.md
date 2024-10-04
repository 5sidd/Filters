# Filters

### Overview
This is a Python project that implements some of the important Computer Vision filtering techniques. The implemented techniques include Mean/Box Filtering, Gaussian Filtering, and Non-Local Means Filtering. First, it takes an original image and generates a new image with noise. The filters are then applied on the noisy image. After applying one of these filters on the noisy image, the results can be found in the results subfolder. The Mean/Box and the Guassian Filtering techniques are implemented in `linear_filtering.py`, the Bilateral Filtering technique is implemented in `bilateral_filtering.py`, and the Non-Local Means Filtering technique is implemented in `nlm_filtering.py`.

### How to Run
1. Install numpy: `pip install numpy`
2. Install opencv-python: `pip install opencv-python`
3. To apply a filter on the noisy image: `python <filter file name>`

### Original Image
![Screenshot 2024-10-04 153209](https://github.com/user-attachments/assets/26bd433e-cc32-413a-b2d7-18f0c5d6b4cb)

### Noisy Image
![image](https://github.com/user-attachments/assets/b8eda22e-1a7e-43e0-847c-ee0b0df3f77e)

### Mean/Box Filter
![image](https://github.com/user-attachments/assets/d271b5fd-c05a-4061-a23a-1608989c946f)

### Guassian Filter
![image](https://github.com/user-attachments/assets/a0e28d83-c003-47f8-a2e7-b9f6aed337bf)

### Bilateral Filter
![image](https://github.com/user-attachments/assets/80cf775e-32b9-4083-8244-e805173f9c7e)

### Non-Local Means Filter
![image](https://github.com/user-attachments/assets/b22b5620-63f6-4d23-9168-82c0e2b8e64f)

Note that the Non-Local Means filter will take around 4-5 minutes to complete its execution.
