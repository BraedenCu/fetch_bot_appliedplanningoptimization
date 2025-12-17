
# Go On Bot

## Install dependencies

```bash
# install uv (mac)
curl -LsSf https://astral.sh/uv/install.sh | sh

# setup environment
uv venv --python 3.10 && source .venv/bin/activate

# install libraries
uv pip install -e .

uv pip install mujoco

# setting up sdk
uv pip install xarm-python-sdk

# mujoco ufactory lite 6
pip install github-clone
ghclone https://github.com/google-deepmind/mujoco_menagerie/tree/bf756430b615819654b640f321c71ba5c3ebeef8/ufactory_lite6

# setting up robot arm
# in one terminal
watch -n1 "ip addr show enp7s0"

# in other terminal
ip addr show enp7s0
sudo ip link set enp7s0 up

# add to subnet
sudo ip addr add 192.168.1.20/24 dev enp7s0
ping 192.168.1.161

# wipe wifi modifications
sudo nmcli connection delete $(nmcli -t -f UUID connection show)
```

## Documentation

- [MuJoCo uFactory Lite6](https://github.com/google-deepmind/mujoco_menagerie/tree/main/ufactory_lite6)
- [xArm Python SDK](https://github.com/xArm-Developer/xArm-Python-SDK)
- [uFactory Lite6 Manual](https://cdn.robotshop.com/media/U/Ufa/RB-Ufa-32/pdf/ufactory-lite-6-user-manual.pdf)
