from __future__ import print_function
import filecmp
import os.path
import sys

compare_file_data = True

def compare_dir_trees(dir1, dir2, compare_file_data, output):

    def compare_dirs(dir1, dir2):
        dirs_cmp = filecmp.dircmp(dir1, dir2)
    
        if dirs_cmp.left_only:
            print("files or subdirs only in %s : %s " % (dir1, dirs_cmp.left_only),
                  file=output)

        if dirs_cmp.right_only:
            print("files or subdirs only in %s : %s " % (dir2, dirs_cmp.right_only),
                  file=output)

        if compare_file_data and dirs_cmp.diff_files:
            print("different files in %s : %s " % (dir1, dirs_cmp.diff_files),
                  file=output)
        
        for common_dir in dirs_cmp.common_dirs:
            new_dir1 = os.path.join(dir1, common_dir)
            new_dir2 = os.path.join(dir2, common_dir)
            compare_dir(new_dir1, new_dir2)

    compare_dirs(dir1, dir2)

dir_a = 'build/server/'
dir_b = 'server/'

print("Compare dirs %s and %s" % (dir_a, dir_b))
compare_dir_trees(dir_a, dir_b, compare_file_data, sys.stdout)

