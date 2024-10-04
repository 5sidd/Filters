import cv2
import numpy as np
import math

def nlm_filtering(
    img: np.uint8,
    intensity_variance: float,
    patch_size: int,
    window_size: int,
) -> np.uint8:

    img = img / 255
    img = img.astype("float32")
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image

    sizeX, sizeY = img.shape
    def out_of_bounds(r, c):
        if r >= 0 and r < sizeX and c >= 0 and c < sizeY:
            return False
        return True
    
    dp = {}
    # compute ssd given centers of patch p and patch q
    def compute_ssd(px, py, qx, qy):
        if (px, py, qx, qy) in dp:
            return dp[(px, py, qx, qy)]
        res = 0
        for k in range(-1 * (patch_size // 2), (patch_size // 2) + 1):
            for l in range(-1 * (patch_size // 2), (patch_size // 2) + 1):
                p_pixel = 0 # f[px + k][py + l]
                q_pixel = 0 # f[qx + k][qy + l]
                if out_of_bounds(px + k, py + l) is False:
                    p_pixel = img[px + k][py + l]
                if out_of_bounds(qx + k, qy + l) is False:
                    q_pixel = img[qx + k][qy + l]
                difference = p_pixel - q_pixel
                squared_difference = math.pow(difference, 2)
                res += squared_difference
        dp[(px, py, qx, qy)] = res
        dp[(qx, qy, px, py)] = res
        return res
      
    # compute weight given ssd of p, q --> w(p, q)
    def compute_wpq(ssd):
        numerator = ssd
        denominator = 2 * intensity_variance
        exponent = -1 * (numerator / denominator)
        return math.exp(exponent)

    # find h(p)  
    # p parameter will always be in bounds       
    def compute_hp(px, py):
        normalization = 0
        res = 0
        for r in range(-1 * (window_size // 2), (window_size // 2) + 1):
            for c in range(-1 * (window_size // 2), (window_size // 2) + 1):
                qx, qy = (px + r), (py + c) # pixel q coordinates
                q_val = 0 # value of pixel q --> f(q)
                if out_of_bounds(qx, qy) is False:
                    q_val = img[qx][qy]
                pq_ssd = compute_ssd(px, py, qx, qy) # SSD(p, q)
                w_pq = compute_wpq(pq_ssd) # w(p, q)
                res += (w_pq * q_val)
                normalization += (w_pq)
        return (1 / normalization) * res
    
    for r in range(sizeX):
        for c in range(sizeY):
            hp = compute_hp(r, c)
            img_filtered[r][c] = hp
                            
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
    intensity_variance = 1
    patch_size = 5 # small image patch size
    window_size = 15 # serach window size
    img_bi = nlm_filtering(img_noise, intensity_variance, patch_size, window_size)
    cv2.imwrite('results/im_nlm.png', img_bi)
    