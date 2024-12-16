import os
from moviepy.editor import VideoFileClip
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser("", add_help=False)
    parser.add_argument("--input", help="input dir", required=True, type=str)
    return parser.parse_args()

if __name__ == "__main__":
    # 指定视频文件夹路径
    args = parse_args()
    video_folder_path = args.input

    # 遍历文件夹中的所有文件
    for filename in tqdm(os.listdir(video_folder_path)):
        if filename.endswith(('.mp4', '.avi', '.mov')):  # 添加你需要转换的视频格式
            # 获取视频文件的完整路径
            video_path = os.path.join(video_folder_path, filename)
            
            # 加载视频
            clip = VideoFileClip(video_path)
            
            # 将视频转换为GIF
            clip.write_gif(video_path + '.gif')