import os


def is_path_exist(create, *paths):
    for path in paths:
        if not os.path.exists(path):
            if create:
                os.mkdir(path)
            else:
                raise FileNotFoundError("路径："+path+"不存在！")
