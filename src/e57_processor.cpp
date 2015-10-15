#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
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
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/common/transforms.h>

typedef pcl::PointXYZI PointXYZ;
typedef pcl::Normal PointNormal;
typedef pcl::PointCloud<PointXYZ> CloudXYZ;
typedef pcl::PointCloud<PointNormal> CloudNormal;

void process_cloud(optional<float> leaf_size, optional<uint32_t> kNN, CloudXYZ::Ptr& cloud, CloudNormal::Ptr& normal_cloud);
void read_scan(e57::Reader& reader, uint32_t scan_index, e57::RigidBodyTransform& scanTransform, CloudXYZ::Ptr& cloud, CloudNormal::Ptr& normal_cloud);
void write_scan(e57::Writer& writer, CloudXYZ::Ptr cloud, CloudNormal::Ptr normal_cloud, const e57::RigidBodyTransform& scanTransform);

typedef std::vector<double> coords_t;
typedef std::tuple<coords_t, coords_t, coords_t> point_data_t;


int main (int argc, char const* argv[]) {
    std::string            fileIn;
    std::string            fileOut;
    std::string            prefixOut;
    float                  leafSize;
    uint32_t               kNN;
    std::vector<uint32_t>  subset;

    po::options_description desc("E57 processor command line options");
    desc.add_options()
        ("help,h",   "Help message")
        ("input,i",  po::value<std::string>(&fileIn) ->required(), "Input E57 File")
        ("output,o", po::value<std::string>(&fileOut), "Output E57 File (optionally use --prefix instead)")
        ("prefix,p", po::value<std::string>(&prefixOut), "Prefix to use for output files")
        ("leaf,l",   po::value<float>(&leafSize)->default_value(0.0), "Leaf Size for subsampling (default is 0; size <= 0 means no subsampling)")
        ("kNN,k",    po::value<uint32_t>(&kNN)->default_value(12), "Number of nearest neighbors to use for normal estimation (value of 0 means no normal estimation)")
        ("subset,s", po::value<std::vector<uint32_t>>(&subset), "Optional scan index subset to extract")
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
        
        if (!subset.size()) {
            subset.resize(scanCount);
            std::iota(subset.begin(), subset.end(), 0);
        }

        for (uint32_t scan_index : subset) {
            std::cout << "scan: " << (scan_index+1) << " / " << scanCount << "\n";
            
            CloudXYZ::Ptr cloud;
            CloudNormal::Ptr normal_cloud;
            
            e57::RigidBodyTransform scanTransform;
            read_scan(reader, scan_index, scanTransform, cloud, normal_cloud);

            optional<float> ls = boost::make_optional(leafSize > 0.f, leafSize);
            optional<uint32_t> k = boost::make_optional(kNN > 0, kNN);
            process_cloud(ls, k, cloud, normal_cloud);

            write_scan(writer_xyz, cloud, normal_cloud, scanTransform);
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
    CloudNormal::Ptr normal_cloud(new CloudNormal());
    pcl::NormalEstimationOMP<PointXYZ, PointNormal> ne;
    ne.setInputCloud(cloud);
    ne.setKSearch(k);
    ne.setViewPoint(0.f, 0.f, 0.f);
    ne.compute(*normal_cloud);

    return normal_cloud;
}

void process_cloud(optional<float> leaf_size, optional<uint32_t> kNN, CloudXYZ::Ptr& cloud, CloudNormal::Ptr& normal_cloud) {
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
        CloudNormal::Ptr subset_normal_cloud(new CloudNormal(*normal_cloud, subset));
        normal_cloud = subset_normal_cloud;
    }

    if (kNN) {
        normal_cloud = compute_normals(cloud, kNN.get());
    }
}

void read_scan(e57::Reader& reader, uint32_t scan_index, e57::RigidBodyTransform& scanTransform, CloudXYZ::Ptr& cloud, CloudNormal::Ptr& normal_cloud) {
    e57::Data3D header;
    reader.ReadData3D(scan_index, header);
    int64_t nColumn = 0, nRow = 0, nPointsSize = 0, nGroupsSize = 0, nCounts = 0; bool bColumnIndex = 0;
    reader.GetData3DSizes( scan_index, nRow, nColumn, nPointsSize, nGroupsSize, nCounts, bColumnIndex);

    int64_t n_size = (nRow > 0) ? nRow : 1024;

    double *data_x = new double[n_size], *data_y = new double[n_size], *data_z = new double[n_size], *data_nx = new double[n_size], *data_ny = new double[n_size], *data_nz = new double[n_size];
    auto block_read = reader.SetUpData3DPointsData(scan_index, n_size, data_x, data_y, data_z, NULL, NULL, NULL, NULL, NULL, NULL, NULL, data_nx, data_ny, data_nz);

    unsigned long size = 0;
    cloud = CloudXYZ::Ptr(new CloudXYZ());
    normal_cloud = CloudNormal::Ptr(new CloudNormal());
    while((size = block_read.read()) > 0) {
        for(unsigned long i = 0; i < size; i++) {
            PointXYZ p_pcl;
            p_pcl.getVector3fMap() = Eigen::Vector3f(
                static_cast<float>(data_x[i]),
                static_cast<float>(data_y[i]),
                static_cast<float>(data_z[i])
            );
            cloud->push_back(p_pcl);
            
            double nx = 0.0, ny = 0.0, nz = 0.0;
            if (header.pointFields.sphericalRangeField && header.pointFields.sphericalAzimuthField && header.pointFields.sphericalElevationField) {
                nx = data_nx[i];
                ny = data_ny[i];
                nz = data_nz[i];
            }
            
            PointNormal n_pcl;
            n_pcl.getNormalVector3fMap() = Eigen::Vector3f(
                static_cast<float>(nx),
                static_cast<float>(ny),
                static_cast<float>(nz)
            );
            normal_cloud->push_back(n_pcl);
        }
    }
    block_read.close();

    delete [] data_x;
    delete [] data_y;
    delete [] data_z;
    delete [] data_nx;
    delete [] data_ny;
    delete [] data_nz;

    // Remember the original scan pose for later writing.
    scanTransform = header.pose;
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
    // FIXME: necessary?
    //header.pointFields.cartesianInvalidStateField = true;
    if (has_normals) {
        header.pointFields.sphericalRangeField = true;
        header.pointFields.sphericalAzimuthField = true;
        header.pointFields.sphericalElevationField = true;
        // FIXME: necessary?
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

void write_scan(e57::Writer& writer, CloudXYZ::Ptr cloud, CloudNormal::Ptr normal_cloud, const e57::RigidBodyTransform& scanTransform) {
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
    
    // Copy the original scan pose.
    header.pose = scanTransform;

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
}
