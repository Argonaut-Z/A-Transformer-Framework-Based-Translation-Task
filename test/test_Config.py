import sys
import os

# 确保添加绝对路径
project_dir = '/mnt/workspace/translate/'
sys.path.append(project_dir)

from config.config import Config

if __name__ == '__main__':
    config = Config()
    print(config.project_dir)
    print(config.train_corpus_file_paths)
