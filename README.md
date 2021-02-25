# sim_nank
 
weightsフォルダを作成して以下のファイルを格納する

https://drive.google.com/drive/folders/1OQqN2wx-pSPBJm7VdvArIO59_arPgRoI?usp=sharing


# DEMO1(jetbot単体）
 
```
roslaunch sim_nank jetbot_controller.launch
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