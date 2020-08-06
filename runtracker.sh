#!/bin/sh

echo "Enter the video path"
read vpath
python /home/cc/nanonets_object_tracking/detection/yolo.py --play_video "True" --video_path "$vpath"
echo "Detection Done"
echo "Start Tracking..."
python /home/cc/nanonets_object_tracking/test_on_video.py --video_path "$vpath"
