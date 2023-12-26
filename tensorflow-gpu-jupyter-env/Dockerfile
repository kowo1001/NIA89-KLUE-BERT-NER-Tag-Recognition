#FROM jupyter/tensorflow-notebook:python-3.10.11
FROM tensorflow/tensorflow:2.13.0-gpu-jupyter
#FROM tensorflow/tensorflow:latest-gpu-jupyter
#FROM tensorflow/tensorflow:latest
#FROM jupyter/tensorflow-notebook:latest
#FROM jupyter/scipy-notebook:python-3.10
#FROM  python:3.10.12

# RUN pip install torch==2.0.1
RUN pip install nvidia-cudnn-cu11==8.6.0.163
RUN pip install transformers
RUN pip install seqeval
RUN pip install pandas

#RUN pip install tensorflow
RUN pip install jupyter
RUN jupyter notebook --generate-config -y
