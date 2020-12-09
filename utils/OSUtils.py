import os


class OSUtils():
    @staticmethod
    def list_file_name(path):
        file_names = [file for file in os.listdir(path)]
        file_names.sort()
        return file_names
