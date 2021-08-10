FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt-get update -y
RUN apt-get -y install openssh-client vim tmux sudo apt-transport-https apt-utils curl git wget lsb-release ca-certificates gnupg gcc g++ pv iftop openmpi-bin openmpi-common libopenmpi-dev
RUN apt-get update -y
RUN apt -y install openjdk-8-jdk
RUN apt -y install libxml-parser-perl
RUN apt -y install zip unzip

ENV MINICONDA_VERSION py37_4.9.2
ENV PATH /opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

RUN pip install pip==9.0.0
RUN pip install ruamel.yaml==0.16 --disable-pip-version-check
RUN pip install --upgrade pip


COPY env.yaml /tmp
RUN conda env create --file /tmp/env.yaml

SHELL ["conda", "run", "-n", "vlt5", "/bin/bash", "-c"]

COPY down.py /tmp
RUN python /tmp/down.py
# RUN python -c "language_evaluation.download('coco')"
# cmd ["python", "-c", "import language_evaluation; language_evaluation.download('coco')"]

