#include <iostream>
#include <fstream>
#include <tuple>
#include <e57/E57Foundation.h>
#include <e57/E57Simple.h>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <boost/none.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
using boost::optional;
using boost::none;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/common/transforms.h>

typedef pcl::PointXYZI PointXYZ;
typedef pcl::Normal PointNormal;
typedef pcl::PointCloud<PointXYZ> CloudXYZ;
typedef pcl::PointCloud<PointNormal> CloudNormal;

std::pair<CloudXYZ::Ptr, CloudNormal::Ptr> process_cloud(CloudXYZ::Ptr cloud, optional<float> leaf_size, optional<uint32_t> kNN);
Eigen::Vector3d read_origin(e57::Reader& reader, uint32_t scan_index);
CloudXYZ::Ptr read_scan(e57::Reader& reader, uint32_t scan_index, const Eigen::Vector3d& offset);
void write_scan(e57::Writer& writer, CloudXYZ::Ptr cloud, CloudNormal::Ptr normal_cloud);

typedef std::vector<double> coords_t;
typedef std::tuple<coords_t, coords_t, coords_t> point_data_t;


int main (int argc, char const* argv[]) {
	std::string  fileIn;
	std::string  fileOut;
	std::string  prefixOut;
    float        leafSize;
    uint32_t     kNN;

	po::options_description desc("Graphene command line options");
	desc.add_options()
		("help,h",  "Help message")
		("input,i",     po::value<std::string>(&fileIn) ->required(), "Input E57 File")
		("output,o",     po::value<std::string>(&fileOut), "Output E57 File (optionally use --prefix instead)")
		("prefix,p",  po::value<std::string>(&prefixOut), "Prefix to use for output files")
		("leaf,l",   po::value<float>(&leafSize)->default_value(0.0), "Leaf Size for subsampling (default is 0; size <= 0 means no subsampling)")
		("kNN,k",        po::value<uint32_t>(&kNN)->default_value(12), "Number of nearest neighbors to use for normal estimation (default is 0; value of 0 means no normal estimation)")
	;

	// Check for required options.
	po::variables_map vm;
	bool optionsException = false;
	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
	} catch (std::exception& e) {
		if (!vm.count("help")) {
			std::cout << e.what() << "\n";
		}
		optionsException = true;
	}
	if (optionsException || vm.count("help")) {
		std::cout << desc << "\n";
		return optionsException ? 1 : 0;
	}

    fs::path pFileIn(fileIn);
    if (!fs::exists(pFileIn)) {
        std::cerr << "File \"" << fileIn << "\" does not exist. Aborting.\n";
        return 1;
    }

    std::string suffixOut = ".e57";
    if (vm.count("prefix") == 0) {
        prefixOut = pFileIn.parent_path().string() + "/" + fs::basename(pFileIn);
    }

    if (vm.count("prefix") && vm.count("output")) {
        std::cout << "Either use --prefix or --output, but not both." << "\n";
        return 1;
    }

    if (vm.count("output")) {
        fs::path pFileOut(fileOut);
        std::string parent_path = pFileOut.parent_path().string();
        if (parent_path == "") parent_path = ".";
        prefixOut = parent_path + "/" + fs::basename(pFileOut);
        suffixOut = pFileOut.extension().string();
    }

	try {
		e57::Reader reader(fileIn);
        e57::E57Root root;
        reader.GetE57Root(root);
        std::string output_file = prefixOut + suffixOut;
        std::cout << "output file: " << output_file << "\n";
		e57::Writer writer_xyz(output_file, root.coordinateMetadata);

		uint32_t scanCount = reader.GetData3DCount();
		uint32_t imgCount = reader.GetImage2DCount();
		std::cout << "number of scans: " << scanCount << "\n";
		std::cout << "number of images: " << imgCount << "\n\n";

        Eigen::Vector3d first_origin = read_origin(reader, 0);

		for (uint32_t scan_index = 0; scan_index < scanCount; ++scan_index) {
			std::cout << "scan: " << (scan_index+1) << " / " << scanCount << "\n";
            CloudXYZ::Ptr cloud = read_scan(reader, scan_index, -first_origin);

            optional<float> ls = boost::make_optional(leafSize > 0.f, leafSize);
            optional<uint32_t> k = boost::make_optional(kNN > 0, kNN);
            CloudNormal::Ptr normal_cloud;
            std::tie(cloud, normal_cloud) = process_cloud(cloud, ls, k);

            write_scan(writer_xyz, cloud, normal_cloud);
		}

        writer_xyz.Close();
	} catch (e57::E57Exception& e) {
        std::cout << "Exception thrown:" << "\n";
		std::cout << e.what() << "\n";
        std::cout << "Context:" << "\n";
        std::cout << e.context() << "\n";
	}

	return 0;
}

CloudNormal::Ptr compute_normals(CloudXYZ::Ptr cloud, uint32_t k) {
    for (auto& p : *cloud) {
        p.getVector3fMap() -= cloud->sensor_origin_.head(3);
    }

    CloudNormal::Ptr normal_cloud(new CloudNormal());
    pcl::NormalEstimation<PointXYZ, PointNormal> ne;
    ne.setInputCloud(cloud);
    ne.setKSearch(k);
    ne.setViewPoint(0.f, 0.f, 0.f);
    ne.compute(*normal_cloud);

    for (auto& p : *cloud) {
        p.getVector3fMap() += cloud->sensor_origin_.head(3);
    }

    return normal_cloud;
}

std::pair<CloudXYZ::Ptr, CloudNormal::Ptr> process_cloud(CloudXYZ::Ptr cloud, optional<float> leaf_size, optional<uint32_t> kNN) {
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

    if (leaf_size) {
        pcl::UniformSampling<PointXYZ> us;
        us.setInputCloud(cloud);
        us.setRadiusSearch(leaf_size.get());
        pcl::PointCloud<int> subsampled_indices;
        us.compute (subsampled_indices);
        std::sort (subsampled_indices.points.begin (), subsampled_indices.points.end ());
        std::vector<int> subset(subsampled_indices.points.begin(), subsampled_indices.points.end());
        CloudXYZ::Ptr subset_cloud(new CloudXYZ(*cloud, subset));
        cloud = subset_cloud;
    }

    CloudNormal::Ptr normal_cloud;
    if (kNN) {
        normal_cloud = compute_normals(cloud, kNN.get());
    }

    return {cloud, normal_cloud};
}

Eigen::Vector3d read_origin(e57::Reader& reader, uint32_t scan_index) {
    e57::Data3D header;
    reader.ReadData3D(scan_index, header);
    return Eigen::Vector3d(header.pose.translation.x, header.pose.translation.y, header.pose.translation.z);
}

CloudXYZ::Ptr read_scan(e57::Reader& reader, uint32_t scan_index, const Eigen::Vector3d& offset) {
    e57::Data3D header;
    reader.ReadData3D(scan_index, header);
    int64_t nColumn = 0, nRow = 0, nPointsSize = 0, nGroupsSize = 0, nCounts = 0; bool bColumnIndex = 0;
    reader.GetData3DSizes( scan_index, nRow, nColumn, nPointsSize, nGroupsSize, nCounts, bColumnIndex);

    int64_t n_size = (nRow > 0) ? nRow : 1024;

    double *data_x = new double[n_size], *data_y = new double[n_size], *data_z = new double[n_size], *intensity = new double[n_size];
    auto block_read = reader.SetUpData3DPointsData(scan_index, n_size, data_x, data_y, data_z, NULL, intensity);

    unsigned long size = 0;
    float imin = header.intensityLimits.intensityMinimum;
    float imax = header.intensityLimits.intensityMaximum;
    std::vector<Eigen::Vector3d> points;
    std::vector<float> intensities;
    uint32_t idx = 0;
    while((size = block_read.read()) > 0) {
        for(unsigned long i = 0; i < size; i++) {
            Eigen::Vector3d p;
            p[0] = data_x[i];
            p[1] = data_y[i];
            p[2] = data_z[i];
            points.push_back(p);
            intensities.push_back((intensity[i]-imin) / (imax-imin));
            //p.intensity = (intensity[i]-imin) / (imax-imin);
            //cloud->push_back(p);
        }
    }
    block_read.close();

    delete [] data_x;
    delete [] data_y;
    delete [] data_z;

    // transform by registration transform
    Eigen::Affine3d registration;
    registration = Eigen::Quaterniond(header.pose.rotation.w, header.pose.rotation.x, header.pose.rotation.y, header.pose.rotation.z);
    registration = Eigen::Translation<double, 3>(header.pose.translation.x + offset[0], header.pose.translation.y + offset[1], header.pose.translation.z + offset[2]) * registration;

    // copy to pointcloud
    CloudXYZ::Ptr cloud(new CloudXYZ());
    for (auto& p : points) {
        p = registration * p;
        PointXYZ p_pcl;
        p_pcl.getVector3fMap() = p.template cast<float>();
        p_pcl.intensity = intensities[idx++];
        cloud->push_back(p_pcl);
    }


    //Eigen::Vector4f translation(header.pose.translation.x + offset[0], header.pose.translation.y + offset[1], header.pose.translation.z + offset[2], 0.f);
    //cloud->sensor_origin_ = translation;
    //Eigen::Quaternionf rotation(header.pose.rotation.w, header.pose.rotation.x, header.pose.rotation.y, header.pose.rotation.z);
    //pcl::transformPointCloud(*cloud, *cloud, translation.head(3), rotation);

    return cloud;
}

void write_scan_data(e57::Writer& writer, e57::Data3D& header, point_data_t& pos_data, point_data_t& normal_data) {
    uint32_t num_points = std::get<0>(pos_data).size();
    if (num_points != std::get<1>(pos_data).size() || num_points != std::get<2>(pos_data).size()) {
        throw std::runtime_error("Exception while writing data: Inconsistent data counts");
    }
    bool has_normals = std::get<0>(normal_data).size() == std::get<0>(pos_data).size();

    header.pointFields.cartesianXField = true;
    header.pointFields.cartesianYField = true;
    header.pointFields.cartesianZField = true;
    //header.pointFields.cartesianInvalidStateField = true;
    if (has_normals) {
        header.pointFields.sphericalRangeField = true;
        header.pointFields.sphericalAzimuthField = true;
        header.pointFields.sphericalElevationField = true;
        //header.pointFields.sphericalInvalidStateField = true;
    }

    int scan_index = writer.NewData3D(header);

    int8_t pos_valid = 1, nrm_valid = has_normals ? 1 : 0;
    e57::CompressedVectorWriter block_write = writer.SetUpData3DPointsData(
        scan_index,
        num_points,
        std::get<0>(pos_data).data(),
        std::get<1>(pos_data).data(),
        std::get<2>(pos_data).data(),
        &pos_valid,
        NULL, NULL, NULL, NULL, NULL, NULL,
        has_normals ? std::get<0>(normal_data).data() : NULL,
        has_normals ? std::get<1>(normal_data).data() : NULL,
        has_normals ? std::get<2>(normal_data).data() : NULL,
        &nrm_valid
    );
    block_write.write(num_points);
    block_write.close();
}

void write_scan(e57::Writer& writer, CloudXYZ::Ptr cloud, CloudNormal::Ptr normal_cloud) {
    // write positions
    e57::Data3D header;
    boost::uuids::random_generator gen;
    boost::uuids::uuid uuid = gen();
    std::stringstream ss_uuid;
    ss_uuid << "{" << uuid << "}";
    header.guid = ss_uuid.str().c_str();

    uint32_t num_points = cloud->size();

    header.pointsSize = num_points * 3;
    header.pointFields.cartesianXField = true;
    header.pointFields.cartesianYField = true;
    header.pointFields.cartesianZField = true;
    header.pose.translation.x = cloud->sensor_origin_[0];
    header.pose.translation.y = cloud->sensor_origin_[1];
    header.pose.translation.z = cloud->sensor_origin_[2];

    // write position data
    bool has_normals = normal_cloud && normal_cloud->size() == cloud->size();
    point_data_t pos_data(coords_t(num_points, 0.0), coords_t(num_points, 0.0), coords_t(num_points, 0.0));
    point_data_t nrm_data(coords_t(has_normals ? num_points : 0), coords_t(has_normals ? num_points : 0), coords_t(has_normals ? num_points : 0));
    uint32_t idx = 0;
    for (const auto& p : *cloud) {
        std::get<0>(pos_data)[idx] = p.x;
        std::get<1>(pos_data)[idx] = p.y;
        std::get<2>(pos_data)[idx] = p.z;
        if (has_normals) {
            std::get<0>(nrm_data)[idx] = normal_cloud->points[idx].normal[0];
            std::get<1>(nrm_data)[idx] = normal_cloud->points[idx].normal[1];
            std::get<2>(nrm_data)[idx] = normal_cloud->points[idx].normal[2];
        }
        ++idx;
    }
    write_scan_data(writer, header, pos_data, nrm_data);

    // write normal data
    //if (!normal_cloud) return;
    //header.pose.translation.x = 0.0;
    //header.pose.translation.y = 0.0;
    //header.pose.translation.z = 0.0;
    //idx = 0;
    //for (const auto& p : *normal_cloud) {
        //data_x[idx] = p.normal[0];
        //data_y[idx] = p.normal[1];
        //data_z[idx] = p.normal[2];
        //++idx;
    //}
    //write_scan_data(writer_normals, header, data_x, data_y, data_z);
}
