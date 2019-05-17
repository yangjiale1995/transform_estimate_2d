#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <iostream>

//计算雅克比矩阵
Eigen::MatrixXf computeJacobi(pcl::PointCloud<pcl::PointXYZI> laser, float yaw)
{
    int cloud_size = laser.points.size();
    Eigen::MatrixXf jacobi(2 * cloud_size, 3);
    for(int i = 0; i < laser.points.size(); i ++)
    {
        Eigen::MatrixXf point_jacobi(2, 3);
        point_jacobi.block<2, 2>(0, 0) = Eigen::Matrix2f::Identity();
        point_jacobi(0, 2) = -sin(yaw) * laser.points[i].x - cos(yaw) * laser.points[i].y;
        point_jacobi(1, 2) = cos(yaw) * laser.points[i].x - sin(yaw) * laser.points[i].y;
        jacobi.block<2, 3>(i * 2, 0) = point_jacobi;
    }
    return jacobi;
}

void matchPreLaser(pcl::PointCloud<pcl::PointXYZI> pre_laser, pcl::PointCloud<pcl::PointXYZI> laser)
{
    //初始化
    float x = 0, y = 0, yaw = 0;

    //构建kd树
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(pre_laser.makeShared());

    //迭代
    for(int iter = 0; iter < 20; iter ++)
    {
        pcl::PointCloud<pcl::PointXYZI> laserCloudOri, coeff;
        
        //构建匹配
        for(int i = 0; i < laser.points.size(); i ++)
        {
            pcl::PointXYZI pointOri = laser.points[i];
            pcl::PointXYZI pointSel;
            pointSel.x = cos(yaw) * pointOri.x - sin(yaw) * pointOri.y + x;
            pointSel.y = sin(yaw) * pointOri.x + cos(yaw) * pointOri.y + y;
            pointSel.z = pointOri.z;

            //kd树查找上一帧中对应的最近点
            std::vector<int> indices;
            std::vector<float> distances;
            kdtree.nearestKSearch(pointSel, 1, indices, distances);
            if(sqrt(distances[0]) > 0.1)
            {
                continue;
            }

            //误差项
            pcl::PointXYZI coe;
            coe.x = pre_laser.points[indices[0]].x - pointSel.x;
            coe.y = pre_laser.points[indices[0]].y - pointSel.y;

            //保存匹配点
            laserCloudOri.push_back(pointOri);
            coeff.push_back(coe);
        }

        //大于2个点才能求解
        if(coeff.points.size() >= 2)
        {
            Eigen::MatrixXf fx(2 * coeff.points.size(), 1);
            for(int i = 0; i < coeff.points.size(); i ++)
            {
                fx(2 * i, 0) = coeff.points[i].x;
                fx(2 * i + 1, 0) = coeff.points[i].y;
            }

            Eigen::MatrixXf jacobi = computeJacobi(laserCloudOri, yaw);
            Eigen::MatrixXf JTJ = jacobi.transpose() * jacobi;
            Eigen::MatrixXf JTF = jacobi.transpose() * fx;
            Eigen::MatrixXf delta = JTJ.ldlt().solve(JTF);

            x += delta(0);
            y += delta(1);
            yaw += delta(2);
        }

    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "test");
    ros::NodeHandle nh;
    pcl::PointCloud<pcl::PointXYZI> pre_laser, laser;
    matchPreLaser(pre_laser, laser);
    return 0;
}

