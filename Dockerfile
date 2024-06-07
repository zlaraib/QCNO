FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3 python3-pip gfortran build-essential libhdf5-openmpi-dev \
                       openmpi-bin pkg-config libopenmpi-dev openmpi-bin libblas-dev \
                       liblapack-dev libpnetcdf-dev git python-is-python3 wget && \
    pip3 install numpy matplotlib h5py sympy scipy jupyter
RUN pip3 install qiskit
RUN pip3 install qiskit-aer qiskit-algorithms qiskit-ibm-provider qiskit-ibm-runtime qiskit-nature qiskit-terra pytket-qiskit
ENV USER=jenkins
ENV LOGNAME=jenkins
