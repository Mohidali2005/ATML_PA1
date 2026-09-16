import numpy as np
from PIL import Image
from skimage.color import lab2rgb,rgb2lab

GRAY_WEIGHTS = np.array([0.2989,0.5870,0.1140],dtype=np.float32)

def resize_224(img,size=224):
    """
    This function resizes a PIL image to a square of the given size
    Every backbone in this task starts from this same resized image
    """
    return img.resize((size,size),Image.BICUBIC)

def to_array(img):
    """
    This function turns a PIL image into a numpy array with values
    scaled between zero and one instead of zero and two fifty five
    """
    return np.asarray(img,dtype=np.float32)/255.0

def grayscale(arr):
    """
    This function removes color from an image by turning it into a
    single brightness value per pixel and then copying that value
    across all three channels so the image shape does not change
    """
    luma = (arr*GRAY_WEIGHTS).sum(axis=-1,keepdims=True)
    return np.repeat(luma,3,axis=-1).astype(np.float32)

def lab_stats(images):
    """
    This function takes a list of images and returns the average and
    the spread of their color values in lab space. It is used to build
    a color signature for a whole class of images
    """
    lab = np.stack([rgb2lab(img) for img in images],axis=0)
    mean = lab.reshape(-1,3).mean(axis=0)
    std = lab.reshape(-1,3).std(axis=0)+1e-6
    return mean,std

def class_swapped_color_stats(arr,own_mean,own_std,target_mean,target_std):
    """
    This function keeps the shape and the layout of an image the same
    but shifts its color values so they match another class color
    signature instead of its own. This is done in lab space because
    lab space separates brightness from color much better than rgb
    """
    lab = rgb2lab(arr)
    normalized = (lab-own_mean)/own_std
    swapped = normalized*target_std+target_mean
    swapped[...,0] = np.clip(swapped[...,0],0,100)
    rgb = lab2rgb(swapped)
    return np.clip(rgb,0.0,1.0).astype(np.float32)

def translate(arr,shift_y,shift_x,max_disp):
    """
    This function moves the content of an image up down left or right
    by a chosen number of pixels. The empty space this creates is
    filled by reflecting the image at its border instead of using
    black pixels. A shift of zero returns the image unchanged
    """
    d = max_disp
    if d == 0:
        return arr.copy()
    padded = np.pad(arr,((d,d),(d,d),(0,0)),mode="reflect")
    h,w = arr.shape[:2]
    top = d-shift_y
    left = d-shift_x
    return padded[top:top+h,left:left+w].copy()

def patch_shuffle(arr,grid_size,rng):
    """
    This function cuts an image into a grid of square patches and then
    rearranges those patches into a new random order. The order is
    checked so it is never the same as the original layout
    """
    h,w = arr.shape[:2]
    ph,pw = h//grid_size,w//grid_size
    patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            patch = arr[i*ph:(i+1)*ph,j*pw:(j+1)*pw]
            patches.append(patch)

    n = len(patches)
    order = np.arange(n)
    identity = np.arange(n)
    while True:
        rng.shuffle(order)
        if not np.array_equal(order,identity):
            break

    shuffled = np.zeros_like(arr)
    for new_pos in range(n):
        old_pos = order[new_pos]
        i,j = divmod(new_pos,grid_size)
        shuffled[i*ph:(i+1)*ph,j*pw:(j+1)*pw] = patches[old_pos]
    return shuffled

def build_color_stats(images,labels):
    """
    This function computes a lab color signature for every class using
    the given images and labels. The result is a dictionary from a
    class id to a mean and a standard deviation
    """
    stats = {}
    for class_id in np.unique(labels):
        class_images = images[labels==class_id]
        mean,std = lab_stats(class_images)
        stats[str(int(class_id))] = {"mean":mean.tolist(),"std":std.tolist()}
    return stats

def build_grayscale_batch(images):
    """
    This function applies the grayscale transform to a whole array of
    images and returns the new array
    """
    return np.stack([grayscale(img) for img in images],axis=0)

def build_color_swap_batch(images,labels,color_stats,rng):
    """
    This function swaps the color statistics of every image with the
    statistics of a different randomly chosen class. The target class
    is chosen fresh for every image using the given random generator
    """
    all_classes = sorted(int(c) for c in color_stats.keys())
    out = np.zeros_like(images)
    for i in range(len(images)):
        own_class = int(labels[i])
        other_classes = [c for c in all_classes if c != own_class]
        target_class = int(rng.choice(other_classes))
        own_mean = np.array(color_stats[str(own_class)]["mean"])
        own_std = np.array(color_stats[str(own_class)]["std"])
        target_mean = np.array(color_stats[str(target_class)]["mean"])
        target_std = np.array(color_stats[str(target_class)]["std"])
        out[i] = class_swapped_color_stats(images[i],own_mean,own_std,target_mean,target_std)
    return out

def build_translation_variant(images,displacement,direction,max_disp):
    """
    This function builds one translated version of a whole array of
    images for the given displacement and direction
    """
    direction_shift = {
        "up":(-1,0),
        "down":(1,0),
        "left":(0,-1),
        "right":(0,1),
    }
    shift_y,shift_x = direction_shift[direction]
    return np.stack(
        [translate(img,shift_y*displacement,shift_x*displacement,max_disp) for img in images],
        axis=0,
    )

def build_patch_shuffle_batch(images,grid_size,rng):
    """
    This function applies one random non identity patch shuffle to
    every image in the given array
    """
    return np.stack([patch_shuffle(img,grid_size,rng) for img in images],axis=0)
