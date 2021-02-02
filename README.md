# sim_nank
 
weightsフォルダを作成して以下のファイルを格納する

best_model_resnet18.pth

best_steering_model_xy.pth

ssd_mobilenet_v2_coco.engine
 
# DEMO1(jetbot単体）
 
```
roslaunch sim_nank object_following.launch
roslaunch sim_nank road_following.launch
roslaunch sim_nank avoidance.launch
roslaunch sim_nank jetbot_controll.launch
```

# DEMO2(jetbotとサーバー）

jetbot

```
rosrun sim_nank controll.py
```


サーバー

```
roslaunch sim_nank jetbot_move.launch
```

# DEMO3(gazebo）・・・hp
```
roslaunch sim_nank sim_move.launch
```

