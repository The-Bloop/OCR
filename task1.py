"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np

def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=5000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments
    
    enrollment(characters)
    
    CCL_dict, results = detection(test_img)
    
    results = recognition(CCL_dict,test_img,results)

    return results
    
    #raise NotImplementedError

def enrollment(characters):
    """ 
    Args:
        characters = list of characters along with name for each character.
    Returns:
        None
    Description:
        Calculates SIFT descriptors for each character from characters.
    """
    # TODO: Step 1 : Your Enrollment code should go here.  
    all_dict = {}
    for i in range(len(characters)):
        name = characters[i][0]
        tup = characters[i][1]
        array = np.array(tup)
        
        thresh_img = binarize(array,100)
        
        val_array = np.where(thresh_img==0)
        bb_max = np.amax(val_array,1)
        bb_min = np.amin(val_array,1)
        
        new_array = thresh_img[bb_min[0]:(bb_max[0] + 1),bb_min[1]:(bb_max[1] + 1)]
        
        hist = compute_sift(new_array)
        all_dict[name] = hist.tolist()
        
    all_json = json.dumps(all_dict) 
    with open('characters_json.json', 'w') as outfile:
        json.dump(all_json, outfile)          
    
    #raise NotImplementedError
            

def detection(img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        img = 2D array
    Returns:
        CCL_dict = Dictionary of SIFT descriptor for Connected Components
        results = list in output format
    Description:
        Detects Characters in components by using Connected Components and calculates SIFT descriptors for each character.
    """
    # TODO: Step 2 : Your Detection code should go here.
    thresh_img = binarize(img,127)
    
    u,v = thresh_img.shape
    
    cc_array = np.zeros((u,v))
    
    label = 0    
    for i in range(0,u):
        top = i
        if(i!=0):
            top = i-1
        for j in range(0,v):
            left = j
            if(j!=0):
                left = j-1
            top_val = 0
            left_val = 0
            if(thresh_img[i][j] == 255):
                cc_array[i][j] = 0
            else:
                if(thresh_img[i][left] == thresh_img[i][j]):
                    left_val = cc_array[i][left]
                else:
                    label = label + 1
                    left_val = label
                if(thresh_img[top][j] == thresh_img[i][j]):
                    top_val = cc_array[top][j]
                else:
                    top_val = left_val
                if(left_val <= top_val):
                    cc_array[i][j] = left_val
                else:
                    cc_array[i][j] = top_val
    
    for i in range(1,u):
        top = i - 1
        for j in range(0,v):
            left = j
            if(j!=0):
                left = j-1
            value_new = -1
            value_old = -1
            if((thresh_img[i][j] == thresh_img[top][j]) & (thresh_img[i][j]==thresh_img[i][left]) & (cc_array[i][left]!=cc_array[top][j])):
                if(cc_array[top][j] < cc_array[i][left]):
                    value_new = cc_array[top][j]
                    value_old = cc_array[i][left]
                else:
                    value_new = cc_array[i][left]
                    value_old = cc_array[top][j]
                    
                cc_array[cc_array == value_old] =  value_new
            
    value_array = np.unique(cc_array)
    
    CCL_dict = {}
    results = []
    for i in range(1,len(value_array)):
        cc_array[cc_array == value_array[i]] =  i
        x_array = np.where(cc_array==i)
        bb_max = np.amax(x_array,1)
        bb_min = np.amin(x_array,1)
        rng = bb_max[:] - bb_min[:] + 1
        new_array = thresh_img[bb_min[0]:(bb_max[0] + 1),bb_min[1]:(bb_max[1] + 1)]      
        
        hist = compute_sift(new_array)
        CCL_dict[i] = hist.tolist()
        result = {'bbox':[int(bb_min[1]),int(bb_min[0]),int(rng[1]),int(rng[1])],'name':'UNKNOWN'}
        results.append(result)
        
    return CCL_dict, results
    #raise NotImplementedError

def recognition(CCL_dict, img, results):
    
    """
    Args:
        CCL_dict = Dictionary of SIFT descriptor for Connected Components
        img = 2D array
        results = list in output format
    Returns:
        None
    Description:
        Calculates SIFT descriptors for characters from CCL_dict, then uses NCC for feature matching.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    
    f = open('characters_json.json')
    data = json.load(f)
    all_dict = json.loads(data)
    
    for i in all_dict:        
        char_hist = all_dict[i]
        char_hist = np.array(char_hist)
        values = []
        for j in CCL_dict:
            hist = CCL_dict[j]
            hist = np.array(hist)
            b = comupte_ncc(hist,char_hist)
            values.append(round(b,4))
            if(round(b,4) >= 0.9905):
                results[(j-1)]["name"] = i
#        values = np.array(values)
#        mkn = np.where(values >= 0.9905)
#        mkn = np.array(mkn)
#        print(i,":",mkn.shape)
    return results

    #raise NotImplementedError 

def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)
        
def binarize(img,limit):
    """
    Args:
        img = 2D array
        limit = binarize limit
    Returns:
        returns img with newly assigned values
    """
    shape = img.shape
    for i in range(0,shape[0]):
        for j in range(0,shape[1]):
            if(img[i][j] >= limit):
                img[i][j] = 255
            else:
                img[i][j] = 0
    return img
    
def compute_sift(img):    
    """
    Args:
        img = image or 2D array
    Returns:
        returns SIFT descriptor vector of size 128
    """
    
    new_shape = (60,60)
    new_img = cv2.resize(img,new_shape)
    new_img = cv2.copyMakeBorder(new_img,2,2,2,2,cv2.BORDER_CONSTANT,value=255)

    gx = cv2.Sobel(new_img,cv2.CV_64F,1,0,ksize=1)
    gy = cv2.Sobel(new_img,cv2.CV_64F,0,1,ksize=1)
    g, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    hist_whole = []
    
    for i in range(0,4):
        for j in range(0,4):
            sub_patch = angle[16*i:16*(i+1), 16*j:16*(j+1)]
            hist,bins = np.histogram(sub_patch,8,[0,360])
            hist_whole.append(hist)
    
    hist_whole = np.array(hist_whole)
    hist_whole = np.reshape(hist_whole,(128))
    return hist_whole

def comupte_ncc(h1,h2):
    h1_mean = np.sum(h1)/128
    h2_mean = np.sum(h2)/128
    
    h1_diff = h1[:] - h1_mean
    h2_diff = h2[:] - h2_mean
    
    h_diff = np.dot(h1_diff,h2_diff)
    numerator = np.sum(h_diff)
    
    h1_d2 = np.power(h1_diff,2)
    h2_d2 = np.power(h2_diff,2)
    h1_d2s = np.sum(h1_d2)
    h2_d2s = np.sum(h2_d2)
    denominator = pow(h1_d2s,(1/2)) * pow(h2_d2s,(1/2))
    
    value = numerator/denominator
    return value


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
