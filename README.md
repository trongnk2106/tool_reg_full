# Web Label face REG
## RUN cpu or gpu
- cpu
```
docker build -t web_label_face_reg_cpu:v1 -f Dockerfile_cpu .
docker run -it --name web_label_face_reg_cpu -p 8000:80 -v $(pwd):/test/ web_label_face_reg_cpu:v1 bash
```
- gpu 
```
docker build -t web_label_face_reg_gpu:v1 -f Dockerfile_gpu .
docker run -it --name web_label_face_reg_gpu -p 8000:5000 -v $(pwd):/test/ web_label_face_reg_gpu:v1 bash
```
## docker-compose
```
docker-compose up
```
