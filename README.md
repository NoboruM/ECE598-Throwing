# ECE598-Throwing
## Dependencies
https://github.com/NoboruM/unitree_sdk2_python  
https://github.com/unitreerobotics/unitree_mujoco

For Nobo:
Run this to enable multicast for the "lo" interface. Required for simulation communication
```
sudo ip link set lo multicast on
```
## If cyclonedds not found
```
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
```

# Test Optas Environment
Original repo: https://github.com/cmower/optas  
Currently uses the pybullet simulator instead of mujoco.  
Test file is based off of dual_arm.py  
## Setup:  
```
pip install pyoptas
pip install pybullet
```
## To run:
```
python3 g1_dual_arm.py
```