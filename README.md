# sim_nank
 
weightsフォルダを作成して以下のファイルを格納する

https://drive.google.com/drive/folders/1OQqN2wx-pSPBJm7VdvArIO59_arPgRoI?usp=sharing

# Testing JetBot1（走行）

サーバー

```
roscore
```

jetbot

```
rosrun jetbot_ros jetbot_motors.py
rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "forward"
rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "backward"
rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "left"
rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "right"
rostopic pub /jetbot_motors/cmd_str std_msgs/String --once "stop"
```

# Testing JetBot2（カメラ）

サーバー

```
roscore
rosrun image_view image_view image:=/jetbot_camera/raw
```

jetbot

```
rosrun jetbot_ros jetbot_camera
```

# DEMO1(jetbot単体）
 
```
roslaunch sim_nank jetbot_controller.launch
roslaunch sim_nank avoidance.launch
roslaunch sim_nank road_following.launch
roslaunch sim_nank object_following.launch
```

# DEMO2(jetbotとサーバー）

最初にサーバーでroscoreを立ち上げておく

jetbot

```
roslaunch sim_nank jetbot_client.launch
```


サーバー

```
roslaunch sim_nank jetbot_move.launch
(rosrun image_view image_view image:=/jetbot_camera/rawでカメラの映像が見える)
roslaunch sim_nank avoidance_server.launch
roslaunch sim_nank road_following_server.launch
roslaunch sim_nank object_following_server.launch
```

# DEMO3(gazebo）・・・hp
```
roslaunch sim_nank sim_move.launch
```

# DEMO4(jetbot and server）

jetbot

```
rosrun sim_nank controller.py
```

server

```
roslaunch sim_nank jetbot_controller.launch
```


# DEMO5(whill and server）

whill


```
roslaunch ros_whill ros_whill.launch
```


server

```
roslaunch sim_nank whill_controller.launch
```