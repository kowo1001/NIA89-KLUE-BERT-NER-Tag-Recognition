FROM tensorflow/tensorflow:2.13.0-gpu

# 필요 라이브러리 설치
RUN apt-get update && apt-get -y install sudo vim && \
    pip install nvidia-cudnn-cu11==8.6.0.163 transformers seqeval pandas tensorflow torch tqdm

ARG UNAME=testuser
ARG USERID=1000
ARG GROUPID=1000
RUN echo "Build arg UID=${USERID}"

RUN echo $UNAME 'ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN sudo mkdir -v /saved_model
RUN sudo mkdir -v /dataset
RUN sudo mkdir -v /dataset_nerlabel
RUN sudo mkdir -v /dataset_tagsequence
RUN sudo mkdir -v /app

RUN sudo chmod 777 -R /dataset/
RUN sudo chmod 777 -R /dataset_nerlabel/
RUN sudo chmod 777 -R /dataset_tagsequence/

# 데이터셋 전처리 파일
COPY --chown=$USERID:$GROUPID ./data_preprocessor.py /app/data_preprocessor.py
# 데이터셋 및 학습관련 파일
COPY --chown=$USERID:$GROUPID ./dataset /app/dataset
COPY --chown=$USERID:$GROUPID ./dataset_nerlabel /app/dataset_nerlabel
COPY --chown=$USERID:$GROUPID ./dataset_tagsequence /app/dataset_tagsequence
COPY --chown=$USERID:$GROUPID ./main.py /app/main.py

WORKDIR /app
USER $UNAME
CMD ["tail","-f","/dev/null"]