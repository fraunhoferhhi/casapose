FROM nvcr.io/nvidia/tensorflow:22.06-tf2-py3

RUN apt -y update && apt -y install libsm6 libxrender1 libfontconfig1 libxtst6 libxi6
RUN apt-get -y update && apt-get -y install libgl1 libglfw3-dev libgles2-mesa-dev

RUN pip install --upgrade pip
RUN pip install opencv-python==4.5.5.62
RUN pip install trimesh==3.9.15
RUN pip install pyrender==0.1.39
RUN pip install imgaug==0.4.0
RUN pip install wget==3.2

ENV WORKPATH=/workspace/CASAPose/
ENV DATAPATH=/workspace/data/

# # The following part is not needed if data and source are mounted into the container from outside.
# RUN git clone https://github.com/fraunhoferhhi/casapose $WORKPATH
# WORKDIR $WORKPATH 

# # download pretrained models
# RUN python util_scripts/download_pretrained_models.py

# # download lmo testset into container (add other datasets here if necessary)
# RUN python util_scripts/prepare_data.py -d $DATAPATH -lmo 
# # cleanup temporary files
# RUN rm -rf ${DATAPATH}tmp

WORKDIR $WORKPATH 
CMD ["bash"]