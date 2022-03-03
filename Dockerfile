FROM python:latest

ARG USERNAME="hiroki"
ARG GROUPNAME="user"
ARG UID=1000
ARG GID=1000
ARG PASSWD="passwd"

RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME && \
    echo $USERNAME:$PASSWD | chpasswd && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER $USERNAME

ADD . /home/$USERNAME/code
WORKDIR /home/$USERNAME/code
RUN pip install -r requirements.txt

ENV DISPLAY host.docker.internal:0.0