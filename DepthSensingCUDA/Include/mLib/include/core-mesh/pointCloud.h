#ifndef CORE_MESH_POINTCLOUD_H_
#define CORE_MESH_POINTCLOUD_H_

namespace ml {

template <class FloatType>
class PointCloud {
public:
	PointCloud() {}

	PointCloud(const std::vector < point3d<FloatType>>& points) {
		m_points = points;
	}

	//!conversion from a binary voxel grid
	PointCloud(const BinaryGrid3& grid, FloatType voxelSize = (FloatType)1) {
		for (unsigned int z = 0; z < grid.dimZ(); z++) {
			for (unsigned int y = 0; y < grid.dimY(); y++) {
				for (unsigned int x = 0; x < grid.dimX(); x++) {
					if (grid.isVoxelSet(x,y,z)) {
						point3d<FloatType> p((FloatType)x,(FloatType)y,(FloatType)z);
						m_points.push_back(p * voxelSize);
					}
				}
			}
		}
	}

	PointCloud(PointCloud&& pc) {
		m_points = std::move(pc.m_points);
		m_normals = std::move(pc.m_normals);
		m_colors = std::move(pc.m_colors);
	}
	void operator=(PointCloud&& pc) {
		m_points = std::move(pc.m_points);
		m_normals = std::move(pc.m_normals);
		m_colors = std::move(pc.m_colors);
	}

	bool hasNormals() const {
		return m_normals.size() > 0;
	}
	bool hasColors() const {
		return m_colors.size() > 0;
	}

	void clear() {
		m_points.clear();
		m_normals.clear();
		m_colors.clear();
	}

	bool isConsistent() const {
		bool is = true;
		if (m_normals.size() > 0 && m_normals.size() != m_points.size())	is = false;
		if (m_colors.size() > 0 && m_colors.size() != m_points.size())		is = false;
		return is;
	}

	bool isEmpty() const {
		return m_points.size() == 0;
	}

	std::vector<point3d<FloatType>> m_points;
	std::vector<point3d<FloatType>> m_normals;
	std::vector<point4d<FloatType>> m_colors;
private:
};

typedef PointCloud<float>	PointCloudf;
typedef PointCloud<double>	PointCloudd;

} // namespace ml


#endif