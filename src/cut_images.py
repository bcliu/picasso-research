from PIL import Image
import Image
import argparse
from os import walk
import os

block_dim = 100
whole_img_dim = 224
whole_img_size = (whole_img_dim, whole_img_dim)
paste_at_x = (whole_img_dim - block_dim) / 2
paste_at_y = (whole_img_dim - block_dim) / 2

parser = argparse.ArgumentParser(description='Shrink image sizes with white padding')
parser.add_argument('load_from_path')
parser.add_argument('save_to_path')
# Whether to only take one block from the center of the image so that not many new images will be created
parser.add_argument('--centeronly', action='store_true', required=False)
args = parser.parse_args()

load_from_path = args.load_from_path
save_to_path = args.save_to_path

if not os.path.exists(save_to_path):
    os.makedirs(save_to_path)

for (dirpath, dirnames, filenames) in walk(args.load_from_path):
    for filename in filenames:
        old_file_path = dirpath + '/' + filename
        im = Image.open(old_file_path)

        width = im.size[0]
        height = im.size[1]

        if not args.centeronly:
            num_blocks_horizonal = width / block_dim
            num_blocks_vertical = height / block_dim
            
            cropped_images_path = save_to_path + '/' + filename
            if not os.path.exists(cropped_images_path):
                os.makedirs(cropped_images_path)

            # Loop through all blocks vertically and horizontally
            for horiz_i in range(num_blocks_horizonal):
                for vert_i in range(num_blocks_vertical):
                    x_from = horiz_i * block_dim
                    y_from = vert_i * block_dim
                    x_to = x_from + block_dim
                    y_to = y_from + block_dim

                    box = (x_from, y_from, x_to, y_to)
                    cropped = im.crop(box)

                    # Put the image in a larger white background
                    new_im = Image.new('RGB', whole_img_size, (255, 255, 255, 255))
                    new_im.paste(cropped, (paste_at_x, paste_at_y))

                    new_file_path = cropped_images_path + '/' + str(vert_i * num_blocks_horizonal + horiz_i) + '.jpg'
                    new_im.save(new_file_path)
        else:
            x_from = (width - block_dim) / 2
            y_from = (height - block_dim) / 2
            x_to = (width + block_dim) / 2
            y_to = (height + block_dim) / 2

            box = (x_from, y_from, x_to, y_to)
            cropped = im.crop(box)

            new_im = Image.new('RGB', whole_img_size, (255, 255, 255, 255))
            new_im.paste(cropped, (paste_at_x, paste_at_y))

            new_file_path = save_to_path + '/cropped' + filename
            new_im.save(new_file_path)
