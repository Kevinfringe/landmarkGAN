'''
    This part of code is intended for duplicate the
    images in jaffe dataset, so that the paired relationship
    can be formed.
    The basic idea is to: duplicate one image to six
    identical image with different names(in order to pair up).
'''
import os
import shutil

train_set_path = "../jaffedbase_official/train_set_jpg/"
train_set_ext_path = "../jaffedbase_official/train_set_final/"

expression_type = {"AN", "DI", "FE", "HA", "NE", "SU", "SA"}

def copy_and_rename_files(source_dir, dest_dir):
    for file_name in os.listdir(source_dir):
        if file_name.endswith(".jpg"):
            # split the file name into prefix, original_exp_type, and suffix
            prefix, original_exp_type, suffix = file_name.split(".", 2)
            for exp_type in expression_type:
                # generate the new file name if the new expression type is different from the original expression type
                if not original_exp_type.startswith(exp_type):
                    new_file_name = f"{prefix}.{original_exp_type}.{exp_type}.{suffix}"
                    # copy the file from the source directory to the destination directory
                    shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, new_file_name))

#  usage
copy_and_rename_files(train_set_path, train_set_ext_path)

# file_name = "KA.AN1.39.jpg"
# prefix, original_exp_type, suffix = file_name.split(".", 2)
# print(original_exp_type)
# print(prefix)
# print(suffix)
