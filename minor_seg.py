from ultralytics import YOLO
import numpy as np

model=YOLO("segmentn.pt")
source="/home/rockingstarniraj/Desktop/minor_cv/minor_dataset/Minor_dataset.v1i.yolov8/test/images/my_photo-168_jpg.rf.d0b319909b50c1a3d7f4df11876e80d7.jpg"
results=model.predict(source=source,conf=0.25,imgsz=640)

for result in results:
  result.show()
  result.save(filename='/home/rockingstarniraj/Desktop/minor_cv/runs/segment/predict7/result3.jpg')

intrinsic=np.array([
    [1.4416e+03,      0,            581.3485], 
     [0,              1.4267e+03,   390.2253],
      [0,             0,            1]
])
extrinsic=np.array([
    [-0.0000000, -0.0000000,      1.0000000  ,    3],
    [0.8191521,     0.5735765  ,   0.0000000,  4],
    [-0.5735765,      0.8191521   ,   -0.0000000,  5],
    [0,         0,                     0,                   1],
])
# [  0.9659258,  0.0000000,  0.2588190;
#    0.0000000,  1.0000000,  0.0000000;
#   -0.2588190,  0.0000000,  0.9659258 ]

# [  0.5540323, -0.7912401,  0.2588190;
#   0.8191521,  0.5735765,  0.0000000;
#  -0.1484525,  0.2120122,  0.9659258 ]

# [ -0.0000000, -0.0000000,  1.0000000;
#    0.8191521,  0.5735765,  0.0000000;
#   -0.5735765,  0.8191521, -0.0000000 ]
inverse_intrinsic=np.linalg.inv(intrinsic)
inverse_extrinsic=np.linalg.inv(extrinsic)


for r in results:
  masks=r.masks
  for mask in masks:
    segmentation_2d_point=np.array(mask.xy[0])
    print((segmentation_2d_point.shape)[0])

    segmentation_hom_point_list=[]

    for i in range((segmentation_2d_point.shape)[0]):
      segmentation_hom_point=np.append(segmentation_2d_point[i],1)
      segmentation_hom_point_list.append(segmentation_hom_point)
    
    homogenous_point=np.array(segmentation_hom_point_list)
    #print(homogenous_point[150])
    transformation_3d_point=[]

    homogenous_point = homogenous_point.T
    #print(homogenous_point[ :,150].shape)
    new_trans_point=inverse_intrinsic @ homogenous_point
    #print(new_trans_point)
    #print(new_trans_point[:,150])
    
    ones_array=[]
    for i in range((segmentation_2d_point.shape)[0]):
      ones_array.append(1)
    ones_array=np.array(ones_array)
    new_trans_point=np.vstack((new_trans_point,ones_array))
    #print(new_trans_point)
    #print(new_trans_point[:,150])
    

    world_point=inverse_extrinsic @ new_trans_point
    # print(world_point)
    # print(world_point[:,170])
    
    drop_row=3
    pseudo_depth=np.delete(world_point,drop_row,axis=0 )
    pseudo_depth=pseudo_depth.T
    #print(pseudo_depth)
    origin_point_of_camera=np.array([0,0,0,1])
    origin_in_real_world= inverse_extrinsic @ origin_point_of_camera
    #print(origin_in_real_world)
    j=2
    coordinate_in_ground_plane=np.array([1,1,1])

    for i in range((segmentation_2d_point.shape)[0]):
      #for j in range(3,0,-1):
      alpha=(-4-origin_in_real_world[j])/pseudo_depth[i][j]
      y_real=origin_in_real_world[j-1]+alpha*pseudo_depth[i][j-1]
      x_real=origin_in_real_world[j-2]+alpha*pseudo_depth[i][j-2]
      x_real=float(x_real)*255
      y_real=float(y_real)*255
      #print(x_real)
      #print(y_real)
      temporary_array= np.array([x_real,y_real,-4])
      print(temporary_array)
      #coordinate_in_ground_plane=np.stack(temporary_array)
    
    #coordinate_in_ground_plane=np.delete(coordinate_in_ground_plane,0,axis=0)

    #print(coordinate_in_ground_plane)









    

    

    
    



    # back_point=intrinsic @ new_trans_point
    # print(back_point)
    # print(back_point)    

    # if(homogenous_point.shape==back_point.shape):
    #   print("you are going right")
    # else:
    #   print("you are wrong")
    
    # for i in range(homogenous_point.shape[1]):
    #    new_gen_point=inverse_intrinsic*homogenous_point[:]
    # #   transformation_3d_point.append(new_gen_point)
    
    # transformation_3d_point=np.array(transformation_3d_point)
    # print(transformation_3d_point[0])


    # new_mask=mask.data
    # print(new_mask)

