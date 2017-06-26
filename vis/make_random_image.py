from utils import initial_loading, localize_image_path
import numpy as np
from PIL import Image

def make_random_image(tile_width):
    # we only need fpaths
    (fpaths, coords, kmeans) = initial_loading()
   
    # generate tile_width**2 random numbers
    indices = [np.random.randint(len(fpaths)) for i in range(tile_width**2)]

    im = make_im_from_indices(indices, tile_width, fpaths)

    im.save('random_image.png', 'PNG')

def make_im_from_indices(indices, tile_width, fpaths):
    image_tile = []
    
    # create tile_width x tile_width array
    for i in range(tile_width):
        k = i*tile_width
        image_tile.append([Image.open(localize_image_path(fpaths[j])) for j in indices[k:k+tile_width]])
    
    im_size = image_tile[0][0].size
    total_width,total_height = (len(image_tile)*im_size[0],len(image_tile[0])*im_size[1])
    
    new_im = Image.new('RGB', (total_width, total_height))
    
    x_offset = 0
    y_offset = 0
    for col in image_tile:
        for row in col:
            new_im.paste(row, (x_offset, y_offset))
            y_offset += im_size[1]
        y_offset = 0
        x_offset += im_size[0]
    
    return new_im

if __name__ == '__main__':
    make_random_image(10)
