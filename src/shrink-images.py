import Image
import argparse
from os import walk

parser = argparse.ArgumentParser(description='Shrink image sizes with white padding')
parser.add_argument('load_from_path')
parser.add_argument('save_to_path')
parser.add_argument('--ratio', default=6, required=False)
args = parser.parse_args()

for (dirpath, dirnames, filenames) in walk(args.load_from_path):
    for filename in filenames:
        old_im = Image.open(dirpath + "/" + filename)

        whole_img_dim = 400
        new_size = (whole_img_dim, whole_img_dim)

        new_im = Image.new("RGB", new_size, (255, 255, 255, 255))
        old_size = old_im.size
        center_height = whole_img_dim / int(args.ratio)
        center_width = int(center_height * 1.0 / old_size[1] * old_size[0])

        old_im = old_im.resize((center_width, center_height), Image.ANTIALIAS)
        old_size = old_im.size

        new_im.paste(old_im, ((new_size[0]-old_size[0])/2,
                              (new_size[1]-old_size[1])/2))

        new_im.save(args.save_to_path + "/" + filename)