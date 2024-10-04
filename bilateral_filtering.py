import cv2
import numpy as np
import math

def bilateral_filtering(
    img: np.uint8,
    spatial_variance: float,
    intensity_variance: float,
    kernel_size: int,
) -> np.uint8:

    img = img / 255
    img = img.astype("float32")
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image
    
    def get_spatial_kernel():
        kernel_weights = np.zeros((kernel_size, kernel_size))
        for r in range(-1 * (kernel_size // 2), (kernel_size // 2) + 1):
            for c in range(-1 * (kernel_size // 2), (kernel_size // 2) + 1):
                numerator = math.pow(r, 2) + math.pow(c, 2)
                denominator = 2 * spatial_variance
                exponent = -1 * (numerator / denominator)
                kernel_weights[r + (kernel_size // 2)][c + (kernel_size // 2)] = math.exp(exponent)
        return kernel_weights
    
    def calculate_intensity_weight(x1, x2):
        x = x1 - x2
        numerator = math.pow(x, 2)
        denominator = 2 * intensity_variance
        exponent = -1 * (numerator / denominator)
        return math.exp(exponent)
    
    spatial_kernel = get_spatial_kernel()
    sizeX, sizeY = img.shape
    
    def out_of_range(r, c):
        if r >= 0 and r < sizeX and c >= 0 and c < sizeY:
            return False
        return True
    
    for r in range(sizeX):
        for c in range(sizeY):
            curr_pixel_val = img[r][c] # f[m][n]
            res = 0 # h[m][n]
            normalization = 0
            for i in range(-1 * (kernel_size // 2), (kernel_size // 2) + 1):
                for j in range(-1 * (kernel_size // 2), (kernel_size // 2) + 1):
                    neighbor_pixel_val = 0 # f[m + k][n + l]
                    offset_r, offset_c = i + (kernel_size // 2), j + (kernel_size // 2)
                    if out_of_range(r + i, c + j) is False:
                        neighbor_pixel_val = img[r + i][c + j]
                    spatial_weight = spatial_kernel[offset_r][offset_c] # g[k][l]
                    intensity_weight = calculate_intensity_weight(curr_pixel_val, neighbor_pixel_val) # r[k][l]
                    res += (spatial_weight * intensity_weight * neighbor_pixel_val)
                    normalization += (spatial_weight * intensity_weight)
            img_filtered[r][c] = (1 / normalization) * res
            
    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered

 
if __name__ == "__main__":
    img = cv2.imread("data/img/butterfly.jpeg", 0) # read gray image
    img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA) # reduce image size for saving your computation time
    cv2.imwrite('results/im_original.png', img) # save image 
    
    # Generate Gaussian noise
    noise = np.random.normal(0,0.6,img.size)
    noise = noise.reshape(img.shape[0],img.shape[1]).astype('uint8')
   
    # Add the generated Gaussian noise to the image
    img_noise = cv2.add(img, noise)
    cv2.imwrite('results/im_noisy.png', img_noise)
    
    # Bilateral filtering
    spatial_variance = 30 # signma_s^2
    intensity_variance = 0.5 # sigma_r^2
    kernel_size = 7
    img_bi = bilateral_filtering(img_noise, spatial_variance, intensity_variance, kernel_size)
    cv2.imwrite('results/im_bilateral.png', img_bi)
    