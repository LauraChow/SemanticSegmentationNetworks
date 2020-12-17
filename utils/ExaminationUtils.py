import os


def is_path_exist(create, *paths):
    for path in paths:
        if not os.path.exists(path):
            if create:
                os.mkdir(path)
            else:
                raise FileNotFoundError("路径："+path+"不存在！")


# def is_file_in_path(file_names, path):
#     is_match = True
#     for file_name in file_names:
#         if not os.path.exists(os.path.join(path, file_name)):
#             is_match = False
#             print("文件：{}不存在！".format(os.path.join(path, file_name)))
#     return is_match
