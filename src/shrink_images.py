from PIL import Image
import argparse
from os import walk
import sys
import os

parser = argparse.ArgumentParser(description='Shrink image sizes with white padding')
parser.add_argument('load_from_path')
parser.add_argument('save_to_path')
parser.add_argument('output_dim')
parser.add_argument('long_edge_pixels')

# Whether to smoothout the edge with transparency gradient
parser.add_argument('--smoothedge', action='store_true', required=False)

args = parser.parse_args()

if not os.path.exists(args.save_to_path):
    os.makedirs(args.save_to_path)

for (dirpath, dirnames, filenames) in walk(args.load_from_path):
    for filename in filenames:
        path = os.path.abspath(os.path.join(dirpath, filename))
        old_im = Image.open(path)

        whole_img_dim = int(args.output_dim)
        new_size = (whole_img_dim, whole_img_dim)

        new_im = Image.new("RGB", new_size, (255, 255, 255))
        old_width, old_height = old_im.size

        if old_width >= old_height:
            center_width = int(args.long_edge_pixels)
            center_height = int(center_width * 1.0 / old_width * old_height)
        else:
            center_height = int(args.long_edge_pixels)
            center_width = int(center_height * 1.0 / old_height * old_width)

        old_im = old_im.resize((center_width, center_height), Image.ANTIALIAS)
        old_im.putalpha(255)
        # Different values this time after resizing
        old_width, old_height = old_im.size

        # Smooth out the edge if toggled
        if args.smoothedge:
            pixels = old_im.load()
            # Create a transparency gradient 20% of center image height
            y_gradient_percentage = 0.2
            x_gradient_percentage = 0.2

            gradient_height = int(old_height * y_gradient_percentage)
            for y in range(gradient_height):
                new_alpha = int(y * 1.0 / gradient_height * 255)
                for x in range(old_width):
                    pixels[x, y] = pixels[x, y][:3] + (new_alpha, )

                    bottom_y = old_height - 1 - y
                    pixels[x, bottom_y] = pixels[x, bottom_y][:3] + (new_alpha, )

            gradient_width = int(old_width * x_gradient_percentage)
            for x in range(gradient_width):
                new_alpha = int(x * 1.0 / gradient_width * 255)
                for y in range(old_height):
                    new_alpha2 = 255 - (255 - pixels[x, y][3] + 255 - new_alpha)
                    if new_alpha2 < 0:
                        new_alpha2 = 0

                    pixels[x, y] = pixels[x, y][:3] + (new_alpha2, )

                    right_x = old_width - 1 - x
                    pixels[right_x, y] = pixels[right_x, y][:3] + (new_alpha2, )

        new_im.paste(old_im, ((new_size[0]-old_width)/2,
                              (new_size[1]-old_height)/2))

        new_im.save(args.save_to_path + "/" + filename)
        
        sys.stdout.write("\x1b[2K\rProcessed: %s" % path)
        sys.stdout.flush()

print '\n'
