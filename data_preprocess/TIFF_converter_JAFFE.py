'''
    This part of code is to convert tiff file into .jpg files.
    ref link: https://stackoverflow.com/questions/28870504/converting-tiff-to-jpeg-in-python
    NOTE: THIS DIDN'T WORK WELL.
'''
import os
from PIL import Image

cur_path = os.getcwd()
train_path = "..\\jaffedbase_official\\train_set\\"

img_path = os.path.join(cur_path, train_path)

print(img_path)


def convertTIFF2JPG():
    for root, dirs, files in os.walk(img_path, topdown=False):
        for name in files:
            print(os.path.join(root, name))
            #print(os.path.splitext(os.path.join(root, name))[0])
            if os.path.splitext(os.path.join(root, name))[1].lower() == ".tiff":
                if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpeg"):
                    print ("A jpeg file already exists for %s" % name)
                # If a jpeg is *NOT* present, create one from the tiff.
                else:
                    outfile = os.path.splitext(os.path.join(root, name))[0] + ".jpeg"
                    try:
                        im = Image.open(os.path.join(root, name))
                        print ("Generating jpeg for %s" % name)
                        im.thumbnail(im.size)
                        im.save(outfile, "JPEG", quality=90)
                    except Exception as e:
                        print(e)

# restore above operations

for file in os.listdir(train_path):
    if file.endswith(".jpeg"):
        os.remove(os.path.join(train_path, file))
        print(os.path.join(train_path, file))
# convertTIFF2JPG()

