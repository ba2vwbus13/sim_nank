# sim_nank
 
weightsフォルダを作成して以下のファイルを格納する

best_model_resnet18.pth

best_steering_model_xy.pth

ssd_mobilenet_v2_coco.engine
 
# DEMO
 
```
roslaunch sim_nank object_following.launch
roslaunch sim_nank road_following.launch
roslaunch sim_nank avoidance.launch
```
