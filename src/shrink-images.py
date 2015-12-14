import Image
import argparse
from os import walk

parser = argparse.ArgumentParser(description='LOL')
parser.add_argument('path')
args = parser.parse_args()

for (dirpath, dirnames, filenames) in walk(args.path):
    for filename in filenames:
        old_im = Image.open(dirpath + "/" + filename)

        whole_img_dim = 400
        new_size = (whole_img_dim, whole_img_dim)

        new_im = Image.new("RGB", new_size)
        center_height = whole_img_dim / 6
        old_size = old_im.size
        old_im = old_im.resize((int(center_height * 1.0 / old_size[1] * old_size[0]), 100), Image.ANTIALIAS)
        old_size = old_im.size

        new_im.paste(old_im, ((new_size[0]-old_size[0])/2,
                              (new_size[1]-old_size[1])/2))

        new_im.save(dirpath + "/" + filename + "-sm.jpg")