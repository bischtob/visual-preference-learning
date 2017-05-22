# default scale for same-size is 25:-1
rm tinder_*
ffmpeg -i tinder-like.png -vf scale=25:-1 tinder_1.png
ffmpeg -i tinder-like.png -vf scale=25:-1 tinder_2.png
ffmpeg -i tinder-like.png -vf scale=25:-1 tinder_3.png
ffmpeg -i tinder-like.png -vf scale=25:-1 tinder_4.png
ffmpeg -i tinder-like.png -vf scale=25:-1 tinder_5.png
