from pathlib import Path
import numpy as np
from .ErrorUtils import error_handler

@error_handler
def write_to_file(file_path:Path, content:str):
    """ファイルに新たに内容を書き込む

    Args:
        file_path (Path): ファイルパス  
        str (str): 書き込む内容
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

@error_handler
def add_to_file(file_path:Path, content:str):
    """ファイルに新たに内容を追加する

    Args:
        file_path (Path): ファイルパス  
        str (str): 書き込む内容
    """
    with open(file_path, "a") as f:
                f.write(content)


@error_handler
def exist_dir(dir_path:Path):
    """ディレクトリとその親が存在するかどうかを調べる

    Args:
        dir_path (Path): ディレクトリパス
    """
    return dir_path.exists() and dir_path.parent.exists()
    

@error_handler
def exist_or_initialize_dir(dir_path:Path):
    """ディレクトリが存在するかを確かめ，存在しないなら追加する
    親まで調べる

    Args:
        dir_path (Path): ディレクトリパス
    """
    if not (exist_dir(dir_path=dir_path)):
        dir_path.mkdir(parents=True)