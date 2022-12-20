from moviepy.editor import *

video = VideoFileClip("./mp4/bonbon.mp4")  # 讀取影片
# output = video.subclip(8, 97)                 # 剪輯影片 ( 單位秒 )
# video = video[0:720, 350:950]
output = video.resize((640, 750))
output.write_videofile("./mp4/bonbon1.mp4", temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
# 輸出影片，注意後方需要加上參數，不然會沒有聲音
print('ok')


# video = VideoFileClip("./mp4/bonbon1.mp4")   # 讀取影片
# audio = video.audio                       # 取出聲音
# audio.write_audiofile("./mp3/bonbon.mp3")         # 輸出聲音為 mp3
# print('ok')
