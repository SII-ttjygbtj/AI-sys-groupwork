#设置继承镜像
FROM ubuntu:latest

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
RUN echo "root:sitdown+1s"|chpasswd

#开放端口
EXPOSE 22
#设置自启动命令
ENTRYPOINT /usr/sbin/sshd -D

