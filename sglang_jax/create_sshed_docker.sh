#!/bin/bash
set -x

IMAGE_NAME="ubuntu:latest"
SSHDED_IMAGE="lwn_sshded_"${IMAGE_NAME//:/_}

DOCKER_NAME="lwn_"${IMAGE_NAME//:/_}
EXPOSED_PORT=26211
ROOT_PASSWD="sitdown+1s"

cat > get_in_${DOCKER_NAME}.sh << EOF
#!/bin/sh
set -x
sshpass -p ${ROOT_PASSWD} ssh -o StrictHostKeyChecking=no -p ${EXPOSED_PORT} root@127.1
EOF

chmod +x get_in_${DOCKER_NAME}.sh


cat > Dockerfile << EOF
#设置继承镜像
FROM ${IMAGE_NAME}

#提供一些作者的信息
MAINTAINER docker_user (user@docker.com)

#下面开始运行更新命令
#RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list
RUN sed -i 's@//ports.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources
RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata

#安装ssh服务
RUN apt-get install -y passwd openssh-server
RUN mkdir /var/run/sshd
RUN sed -i 's/UsePAM yes/UsePAM no/g' /etc/ssh/sshd_config
RUN sed -i "s/.*PermitRootLogin.*/PermitRootLogin yes/" /etc/ssh/sshd_config

#设置初始密码
RUN echo "root:${ROOT_PASSWD}"|chpasswd

#开放端口
EXPOSE 22
#设置自启动命令
ENTRYPOINT /usr/sbin/sshd -D

EOF

docker pull ${IMAGE_NAME}

docker build --no-cache -t ${SSHDED_IMAGE} .

# --shm-size=1g这个flag为了支持nccl。nccl库需要很大的共享内存
# https://github.com/NVIDIA/nccl-tests/issues/143

docker run \
	 -d \
	 --shm-size=1g \
	 -v $(pwd):/data \
	 --name ${DOCKER_NAME} \
	 -p ${EXPOSED_PORT}:22 \
	 ${SSHDED_IMAGE}
