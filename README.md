# ssh_mapping






# Docker

## 1. Install Docker via CLI not snap

### Install using the apt repository

Before you install Docker Engine for the first time on a new host machine, you need to set up the Docker repository. Afterward, you can install and update Docker from the repository.

https://docs.docker.com/engine/install/ubuntu/

1. Set up Docker’s apt repository

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

```

1. nstall the Docker packages.

```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

```

1. Verify that the Docker Engine installation is successful by running the `hello-world` image.

## 2. Install NVIDA GPUs Driver

On Ubuntu:

## Install ubuntu-drivers-common

```bash
sudo apt install ubuntu-drivers-common
```

- run the following cli:

```bash
ubuntu-drivers devices
```

you should see:

```bash
(ssh) ubuntu@l40s-90-gra11:~/project/ssh_mapping$ ubuntu-drivers devices
udevadm hwdb is deprecated. Use systemd-hwdb instead.
udevadm hwdb is deprecated. Use systemd-hwdb instead.
udevadm hwdb is deprecated. Use systemd-hwdb instead.
udevadm hwdb is deprecated. Use systemd-hwdb instead.
ERROR:root:aplay command not found
== /sys/devices/pci0000:00/0000:00:06.0 ==
modalias : pci:v000010DEd000026B9sv000010DEsd00001851bc03sc02i00
vendor   : NVIDIA Corporation
model    : AD102GL [L40S]
driver   : nvidia-driver-535-server-open - distro non-free
driver   : nvidia-driver-535-open - distro non-free
driver   : nvidia-driver-535-server - distro non-free
driver   : nvidia-driver-535 - distro non-free recommended
driver   : xserver-xorg-video-nouveau - distro free builtin

(ssh) ubuntu@l40s-90-gra11:~/project/ssh_mapping$
```

then run the following CLI to install the driver

```bash
sudo apt install nvidia-driver-535
```

The driver is fully installed. Now Nvidia container toolkit is needed.

```bash
sudo docker run hello-world
```

## 3. Install NVIDIA Container Toolkit

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt

The NVIDIA Container Toolkit enables users to build and run GPU-accelerated containers. The toolkit includes a container runtime [library](https://github.com/NVIDIA/libnvidia-container) and utilities to automatically configure containers to leverage NVIDIA GPUs.

![Untitled](https://cloud.githubusercontent.com/assets/3028125/12213714/5b208976-b632-11e5-8406-38d379ec46aa.png)

## Installing with Apt

1. Configure the production repository:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Optionally, configure the repository to use experimental packages:

```bash
sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

1. Update the packages list from the repository:

```bash
sudo apt-get update
```

1. Install the NVIDIA Container Toolkit packages:

```bash
sudo apt-get install -y nvidia-container-toolkit
```