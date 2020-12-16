import os


def list_file_name(path):
    file_names = [os.path.join(path, file) for file in os.listdir(path)
                  if os.path.splitext(file)[-1] in (".jpg", ".png", ".JPG", ".PNG")]
    file_names.sort()
    return file_names
