import argparse
import torch
import sys
import os
parser = argparse.ArgumentParser()
################## for files
parser.add_argument('--ground_path', default=r'D:\Data\3d')
parser.add_argument('--plyfile', default=r'plum.ply')
parser.add_argument('--gripper_rotation_pickle', default=r'D:\Data\3d\R_4_4_30_90.pkl') #4*4 gripper internal 30, rotate 90degree
#for demo
parser.add_argument('--weightcentershowsize', default=5)
parser.add_argument('--voxel_size_05', default=0.5) 
#for physical dimentions
parser.add_argument('--density', default=1000000) # unit g
parser.add_argument('--cup_radius', default=5) # gripper unit mm
parser.add_argument('--cup_points', default=1600) # gripper
parser.add_argument('--gripper_cups_x', default=4) #prediction length cause the stop is not working now
parser.add_argument('--gripper_cups_y', default=4) #prediction length cause the stop is not working now
parser.add_argument('--gripper_cups_d', default=30) #prediction length cause the stop is not working now
#for physical limit
parser.add_argument('--max_load', default=300) # gripper g
parser.add_argument('--max_torque', default=50) # gripper g*mm
parser.add_argument('--max_residual', default=0.01) # gripper g*mm
parser.add_argument('--threhold_residual', default=0.001) # gripper g*mm
parser.add_argument('--earlystop_safty_buffer', default=1) # gripper g*mm
#for digital limit
parser.add_argument('--normal_selection_knn', default=20) 
parser.add_argument('--normal_range', default=0.9) # 
parser.add_argument('--deplan_distance', default=10) # mm
parser.add_argument('--deareax', default=[-100,100]) # area limit
parser.add_argument('--deareay', default=[-100,100]) # 
parser.add_argument('--deareaz', default=[-100,100]) # 
parser.add_argument('--denoise_sampling', default=0.2) # 
parser.add_argument('--denoise_neighb', default=20) # 
parser.add_argument('--denoise_std', default=2.0) # 
#for digital setting
parser.add_argument('--resolution_catch', default=0.5) # 
parser.add_argument('--potential_position_internal', default=10) # 
parser.add_argument('--earlystop', default=True) # 





########################################
parser.add_argument('--cuda_device', default="0,1,2,3") # numbers of cuda
args = parser.parse_args()

