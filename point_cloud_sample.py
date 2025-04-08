#!/usr/bin/env python3

# import roslib
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import pyransac3d as pyrsc
import tf
from geometry_msgs.msg import PointStamped
import sensor_msgs.point_cloud2 as pc2
import pcl_ros
from sklearn.decomposition import PCA
import struct
import ctypes

pub = rospy.Publisher('/output', PointCloud2, queue_size=1)
tmp_pcd_name = "/tmp/tmp_cloud.pcd"

FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
]

TEST_POINTS = [
    [0.3, 0.0, 0.0, 0xff0000],
    [0.0, 0.3, 0.0, 0x00ff00],
    [0.0, 0.0, 0.3, 0x0000ff],
]

def convert_pcl(data):
    header = '''# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH %d
HEIGHT %d
VIEWPOINT 0 0 0 1 0 0 0
POINTS %d
DATA ascii'''

    with open(tmp_pcd_name, 'w') as f:
        f.write(header % (data.width, data.height, data.width*data.height))
        f.write("\n")

        for p in pc2.read_points(data, skip_nans=True):
            f.write('%f %f %f %e' % (p[0], p[1], p[2], p[3]))
            f.write("\n")

        # cloud_list = []
        # for p in pc2.read_points(data, skip_nans=False):
        #     cloud_list.append(p[0])
        #     cloud_list.append(p[1])
        #     cloud_list.append(p[2])
        #     cloud_list.append(p[3])

        f.write("\n")
    pcd = o3d.io.read_point_cloud(tmp_pcd_name)

    return pcd

def publish_pointcloud(output_data, input_data):
    # convert pcl data format
    pc_p = np.asarray(output_data.points)
    pc_c = np.asarray(output_data.colors)
    tmp_c = np.c_[np.zeros(pc_c.shape[1])]
    tmp_c = np.floor(pc_c[:,0] * 255) * 2**16 + np.floor(pc_c[:,1] * 255) * 2**8 + np.floor(pc_c[:,2] * 255) # 16bit shift, 8bit shift, 0bit shift

    pc_pc = np.c_[pc_p, tmp_c]

    # publish point cloud
    output = pc2.create_cloud(Header(frame_id=input_data.header.frame_id), FIELDS , pc_pc)
    pub.publish(output)

def publish_testcloud(input_data):
    # publish point cloud
    output = pc2.create_cloud(Header(frame_id=input_data.header.frame_id), FIELDS , TEST_POINTS)
    pub.publish(output)



def callback(data):
    result_pcl = convert_pcl(data)
    print(result_pcl)
    points = np.asarray(result_pcl.points)
    plano1 = pyrsc.Plane()
    best_eq,best_inliers = plano1.fit(points,0.01)
    plane = result_pcl.select_by_index(best_inliers).paint_uniform_color([1,0,0])
    obb = plane.get_oriented_bounding_box() #obb.center ...[x,y,z]
    obb.color=[0,0,1]
    not_plane = result_pcl.select_by_index(best_inliers,invert=True)
    o3d.visualization.draw_geometries([not_plane,plane,obb])
    # data.header.frame_id
    # publish_pointcloud(result_pcl, data)

    # publish_testcloud(data)

def calc_vector_angle(v0,v1):
    #ベクトルの長さ
    dot = np.dot(v0, v1)
    # ベクトルのノルム
    v0_norm = np.linalg.norm(v0)
    v1_norm = np.linalg.norm(v1)
    return dot/(v0_norm * v1_norm)

if __name__ == "__main__":
    rospy.init_node('listener', anonymous=True)
    # rospy.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/points',
    #                  PointCloud2, callback)
    pca = PCA()
    data = rospy.wait_for_message('/hsrb/head_rgbd_sensor/depth_registered/rectified_points',PointCloud2)
    listener = tf.TransformListener()
    listener.waitForTransform('map', data.header.frame_id,rospy.Time(),rospy.Duration(10))
    (trans,rot) = listener.lookupTransform('map', data.header.frame_id,rospy.Time())
    transform_matrix = listener.fromTranslationRotation(trans,rot)

    xyz = np.array([[0,0,0]])
    rgb = np.array([[0,0,0]])
    int_data = pc2.read_points_list(data,skip_nans=True)
    # int_data = list(gen)
    for x in int_data:
        test = x[3]
        # ビット演算を可能にするため、floatにキャスト
        s = struct.pack('<f', test)
        i = struct.unpack('<l',s)[0]
        # 逆演算で浮動小数点値を取得
        pack = ctypes.c_uint32(i).value
        r = (pack & 0x00FF0000) >> 16
        g = (pack & 0x0000FF00) >> 8
        b = (pack & 0x000000FF)
        # b = pack & 0xff
        # g = (pack >> 8) & 0xff
        # r = (pack >> 16) & 0xff
        
        xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis=0)
        rgb = np.append(rgb,[[r,g,b]], axis=0)
    
    temp = o3d.geometry.PointCloud()
    temp.points = o3d.utility.Vector3dVector(xyz)
    temp.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud("test.ply",temp)
    pcd2 = o3d.io.read_point_cloud("test.ply")
    geometries = []
    geometries.append(pcd2)
    o3d.visualization.draw_geometries(geometries)
    # あとでコメントアウト
    # pcd2 = convert_pcl(data)
    pcd2.transform(transform_matrix)
    pcd2.estimate_normals()


    oboxes = pcd2.detect_planar_patches(
    normal_variance_threshold_deg=60,
    coplanarity_deg=75,
    outlier_ratio=0.75,
    min_plane_edge_length=0,
    min_num_points=0,
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    print("Detected {} patches".format(len(oboxes)))
    normal_line = np.array([0,0,1])

    for obox in oboxes:
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
        mesh.compute_vertex_normals()
        result = pca.fit(np.asarray(mesh.vertices))
        cos = calc_vector_angle(result.components_[2],normal_line)
        if (abs(cos) > 0.9):
            obb_center = obox.get_center()
            if obb_center[2] > 0.3:
                ps = PointStamped()
                ps.header.frame_id = 'map'
                ps.header.stamp = rospy.Time().now()
                ps.point.x = obb_center[0]
                ps.point.y = obb_center[1]
                ps.point.z = obb_center[2]
                listener.waitForTransform('base_link','map',rospy.Time().now(),rospy.Duration(10))
                tf_ps = listener.transformPoint('base_link',ps)
                if tf_ps.point.x < 1:
                    rospy.loginfo(tf_ps)
                    # mesh.paint_uniform_color(obox.color)
                    mesh.paint_uniform_color((np.random.random((3,1))))
                    bbb = mesh.create_coordinate_frame(origin=obox.get_center())
                    geometries.append(bbb)
                    geometries.append(mesh)
                    geometries.append(obox)
    geometries.append(pcd2)
    # aaa = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # geometries.append(aaa)
    o3d.visualization.draw_geometries(geometries)
