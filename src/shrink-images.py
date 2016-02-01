import Image
import argparse
from os import walk

parser = argparse.ArgumentParser(description='Shrink image sizes with white padding')
parser.add_argument('load_from_path')
parser.add_argument('save_to_path')
parser.add_argument('--ratio', default=6, required=False)

# Whether to smoothout the edge with transparency gradient
parser.add_argument('--smoothedge', action='store_true', required=False)

args = parser.parse_args()

for (dirpath, dirnames, filenames) in walk(args.load_from_path):
    for filename in filenames:
        old_im = Image.open(dirpath + "/" + filename)

        whole_img_dim = 400
        new_size = (whole_img_dim, whole_img_dim)

        new_im = Image.new("RGB", new_size, (255, 255, 255, 255))
        old_width, old_height = old_im.size

        center_height = whole_img_dim / int(args.ratio)
        center_width = int(center_height * 1.0 / old_height * old_width)

        old_im = old_im.resize((center_width, center_height), Image.ANTIALIAS)
        old_im.putalpha(255)
        # Different values this time after resizing
        old_width, old_height = old_im.size

        # Smooth out the edge if toggled
        if args.smoothedge:
            pixels = old_im.load()
            # Create a transparency gradient 20% of center image height
            gradient_height = int(old_height * 0.2)
            for y in range(gradient_height):
                new_alpha = int(y * 1.0 / gradient_height * 255)
                for x in range(old_width):
                    print pixels[x, y][:3]
                    pixels[x, y] = pixels[x, y][:3] + (new_alpha, )

        old_im.save('lol.jpg')
        new_im.paste(old_im, ((new_size[0]-old_width)/2,
                              (new_size[1]-old_height)/2))

        new_im.save(args.save_to_path + "/" + filename)