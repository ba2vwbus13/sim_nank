# sim_nank
 
weightsフォルダを作成して以下のファイルを格納する

best_model_resnet18.pth

best_steering_model_xy.pth

mobilenet-v1-ssd-mp-0_675.pth

voc-model-labels.txt

ssd_mobilenet_v2_coco.engine
 
# DEMO1(jetbot単体）
 
```
roslaunch sim_nank jetbot_controll.launch
roslaunch sim_nank avoidance.launch
roslaunch sim_nank road_following.launch
roslaunch sim_nank object_following.launch
```

# DEMO2(jetbotとサーバー）

jetbot

```
roslaunch sim_nank jetbot_client.launch
```


サーバー

```
roslaunch sim_nank jetbot_move.launch
roslaunch sim_nank avoidance_server.launch
roslaunch sim_nank road_following_server.launch
roslaunch sim_nank object_following_server.launch
```

# DEMO3(gazebo）・・・hp
```
roslaunch sim_nank sim_move.launch
```

