import numpy as np

def salt_pepper(image, salt=0.1, pepper=0.1):
    height = image.shape[0]
    width = image.shape[1]
    pertotal = salt + pepper    #总噪声占比
    noise_image = image.copy()
    noise_num = int(pertotal * height * width)
    for i in range(noise_num):
        rows = np.random.randint(0, height-1)
        cols = np.random.randint(0,width-1)
        if(np.random.randint(0,100)<salt*100):
            noise_image[rows][cols] = 255
        else:
            noise_image[rows][cols] = 0
    return noise_image