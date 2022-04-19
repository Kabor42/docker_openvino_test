FROM ubuntu:20.04

ARG PACKAGE=openvino-2022.1.0

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg \
    wget \
    udev

RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
RUN echo "deb https://apt.repos.intel.com/openvino/2022 focal main" > /etc/apt/sources.list.d/intel-openvino-2022.list
RUN rm -rf GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

RUN apt-get update && apt-get install -y --no-install-recommends \
    $PACKAGE
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev
RUN apt-get install -y --no-install-recommends \
    libopencv-core4.2 \
    libopencv-videoio4.2 \
    python3-opencv
RUN apt autoremove -y

RUN /bin/bash -c "source /opt/intel/openvino_2022/setupvars.sh"
RUN echo "source /opt/intel/openvino_2022/setupvars.sh" >> /root/.bashrc

RUN cp /opt/intel/openvino_2022/runtime/3rdparty/97-myriad-usbboot.rules /etc/udev/rules.d/

COPY models /root/models
COPY app /app
COPY startup.sh /app

CMD /bin/bash /app/startup.sh
# CMD /bin/bash
