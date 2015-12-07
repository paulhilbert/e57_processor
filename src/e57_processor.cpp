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

typedef std::vector<double> coords_t;
typedef std::vector<uint16_t> colors_t;
typedef std::tuple<coords_t, coords_t, coords_t> point_data_t;
typedef std::tuple<colors_t, colors_t, colors_t> color_data_t;

void process_cloud(optional<float> leaf_size, optional<uint32_t> kNN, CloudXYZ::Ptr& cloud, CloudNormal::Ptr& normal_cloud, std::vector<Eigen::Vector3f>* colors);
void read_scan(e57::Reader& reader, uint32_t scan_index, e57::RigidBodyTransform& scanTransform, CloudXYZ::Ptr& cloud, CloudNormal::Ptr& normal_cloud, std::vector<Eigen::Vector3f>* color_data = nullptr, bool* color_valid = nullptr);
void write_scan(e57::Writer& writer, CloudXYZ::Ptr cloud, CloudNormal::Ptr normal_cloud, const e57::RigidBodyTransform& scanTransform, std::vector<Eigen::Vector3f>* color_data = nullptr);



int main (int argc, char const* argv[]) {
    std::string            fileIn;
    std::string            fileOut;
    std::string            prefixOut;
    float                  leafSize;
    uint32_t               kNN;
    std::vector<uint32_t>  subset;
    bool                   copy_colors;

    print("yes");

    po::options_description desc("E57 processor command line options");
    desc.add_options()
        ("help,h",   "Help message")
        ("input,i",  po::value<std::string>(&fileIn) ->required(), "Input E57 File")
        ("output,o", po::value<std::string>(&fileOut), "Output E57 File (optionally use --prefix instead)")
        ("prefix,p", po::value<std::string>(&prefixOut), "Prefix to use for output files")
        ("leaf,l",   po::value<float>(&leafSize)->default_value(0.0), "Leaf Size for subsampling (default is 0; size <= 0 means no subsampling)")
        ("kNN,k",    po::value<uint32_t>(&kNN)->default_value(12), "Number of nearest neighbors to use for normal estimation (value of 0 means no normal estimation)")
        ("subset,s", po::value<std::vector<uint32_t>>(&subset), "Optional scan index subset to extract")
        ("colors", "Copy Colors")
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

    copy_colors = vm.count("colors");

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
            std::vector<Eigen::Vector3f> colors;
            bool colors_valid;
            read_scan(reader, scan_index, scanTransform, cloud, normal_cloud, copy_colors ? &colors : nullptr, copy_colors ? &colors_valid : nullptr);

            optional<float> ls = boost::make_optional(leafSize > 0.f, leafSize);
            optional<uint32_t> k = boost::make_optional(kNN > 0, kNN);
            process_cloud(ls, k, cloud, normal_cloud, copy_colors ? &colors : nullptr);

            write_scan(writer_xyz, cloud, normal_cloud, scanTransform, copy_colors ? &colors : nullptr);
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

void process_cloud(optional<float> leaf_size, optional<uint32_t> kNN, CloudXYZ::Ptr& cloud, CloudNormal::Ptr& normal_cloud, std::vector<Eigen::Vector3f>* colors) {
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
        if (colors) {
            std::vector<Eigen::Vector3f> col_subset(subset.size());
            uint32_t idx = 0;
            for (const auto& s_idx : subset) {
                col_subset[idx++] = (*colors)[s_idx];
            }
            *colors = col_subset;
        }
    }

    if (kNN) {
        normal_cloud = compute_normals(cloud, kNN.get());
    }
}

void read_scan(e57::Reader& reader, uint32_t scan_index, e57::RigidBodyTransform& scanTransform, CloudXYZ::Ptr& cloud, CloudNormal::Ptr& normal_cloud, std::vector<Eigen::Vector3f>* color_data, bool* color_valid) {
    e57::Data3D header;
    reader.ReadData3D(scan_index, header);
    int64_t nColumn = 0, nRow = 0, nPointsSize = 0, nGroupsSize = 0, nCounts = 0; bool bColumnIndex = 0;
    reader.GetData3DSizes( scan_index, nRow, nColumn, nPointsSize, nGroupsSize, nCounts, bColumnIndex);

    uint16_t min_r = header.colorLimits.colorRedMinimum;
    uint16_t max_r = header.colorLimits.colorRedMaximum;
    uint16_t min_g = header.colorLimits.colorGreenMinimum;
    uint16_t max_g = header.colorLimits.colorGreenMaximum;
    uint16_t min_b = header.colorLimits.colorBlueMinimum;
    uint16_t max_b = header.colorLimits.colorBlueMaximum;
    float f_min_r = static_cast<float>(min_r);
    float f_rng_r = static_cast<float>(max_r - min_r);
    float f_min_g = static_cast<float>(min_g);
    float f_rng_g = static_cast<float>(max_g - min_g);
    float f_min_b = static_cast<float>(min_b);
    float f_rng_b = static_cast<float>(max_b - min_b);

    int64_t n_size = (nRow > 0) ? nRow : 1024;

    double *data_x = new double[n_size], *data_y = new double[n_size], *data_z = new double[n_size], *data_nx = new double[n_size], *data_ny = new double[n_size], *data_nz = new double[n_size];
    uint16_t* col_r = color_data ? new uint16_t[n_size] : NULL;
    uint16_t* col_g = color_data ? new uint16_t[n_size] : NULL;
    uint16_t* col_b = color_data ? new uint16_t[n_size] : NULL;
    int8_t color_valid_flag = 0;
    auto block_read = reader.SetUpData3DPointsData(scan_index, n_size, data_x, data_y, data_z, NULL, NULL, NULL,
        col_r, col_g, col_b, &color_valid_flag, data_nx, data_ny, data_nz);
    if (color_valid) *color_valid = color_valid_flag != 0;

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
            if (color_data) {
                Eigen::Vector3f c;
                c[0] = (static_cast<float>(col_r[i]) - f_min_r) / f_rng_r;
                c[1] = (static_cast<float>(col_g[i]) - f_min_g) / f_rng_g;
                c[2] = (static_cast<float>(col_b[i]) - f_min_b) / f_rng_b;
                color_data->push_back(c);
            }
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

void write_scan_data(e57::Writer& writer, e57::Data3D& header, point_data_t& pos_data, point_data_t& normal_data, color_data_t& color_data) {
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
    header.pointFields.colorBlueField = true;
    header.pointFields.colorGreenField = true;
    header.pointFields.colorRedField = true;
    header.colorLimits.colorRedMinimum = 0;
    header.colorLimits.colorRedMaximum = 255;
    header.colorLimits.colorGreenMinimum = 0;
    header.colorLimits.colorGreenMaximum = 255;
    header.colorLimits.colorBlueMinimum = 0;
    header.colorLimits.colorBlueMaximum = 255;

    uint16_t max_c = 0;
    for (uint32_t i = 0; i < num_points; ++i) {
        uint16_t c = std::get<0>(color_data)[i];
        if (c > max_c) max_c = c;
        c = std::get<1>(color_data)[i];
        if (c > max_c) max_c = c;
        c = std::get<2>(color_data)[i];
        if (c > max_c) max_c = c;
    }

    int scan_index = writer.NewData3D(header);

    int8_t pos_valid = 1, nrm_valid = has_normals ? 1 : 0, col_valid = 1;
    e57::CompressedVectorWriter block_write = writer.SetUpData3DPointsData(
        scan_index,
        num_points,
        std::get<0>(pos_data).data(),
        std::get<1>(pos_data).data(),
        std::get<2>(pos_data).data(),
        &pos_valid,
        NULL, NULL,
        std::get<0>(color_data).data(),
        std::get<1>(color_data).data(),
        std::get<2>(color_data).data(),
        &col_valid,
        has_normals ? std::get<0>(normal_data).data() : NULL,
        has_normals ? std::get<1>(normal_data).data() : NULL,
        has_normals ? std::get<2>(normal_data).data() : NULL,
        &nrm_valid
    );
    block_write.write(num_points);
    block_write.close();
}

void write_scan(e57::Writer& writer, CloudXYZ::Ptr cloud, CloudNormal::Ptr normal_cloud, const e57::RigidBodyTransform& scanTransform, std::vector<Eigen::Vector3f>* color_data) {
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
    uint16_t max_col = 255;
    color_data_t col_data(colors_t(num_points, max_col), colors_t(num_points, max_col), colors_t(num_points, max_col));
    if (color_data) {
        for (uint32_t i = 0; i < num_points; ++i) {
            std::get<0>(col_data)[i] = static_cast<uint16_t>((*color_data)[i][0] * 255.f);
            std::get<1>(col_data)[i] = static_cast<uint16_t>((*color_data)[i][1] * 255.f);
            std::get<2>(col_data)[i] = static_cast<uint16_t>((*color_data)[i][2] * 255.f);
        }
    }
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
    write_scan_data(writer, header, pos_data, nrm_data, col_data);
}
