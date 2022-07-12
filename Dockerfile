FROM jupyter/minimal-notebook:9b06df75e445

COPY requirements.txt /home/pjk/
WORKDIR /home/pjk

# Local machine files directory
VOLUME /home/slam/coff_EDA 

RUN pip install --upgrade pip &&\
    pip install -r requirements.txt
ADD /model/model_reg_cof.ipynb /model/plot.py /data/dataset_renamed.xlsx /home/pjk/work/ 
