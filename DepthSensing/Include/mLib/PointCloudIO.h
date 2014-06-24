#pragma once



#ifndef _POINT_CLOUD_IO_H_
#define _POINT_CLOUD_IO_H_

namespace ml {

template <class FloatType>
class PointCloudIO {
public:

	static PointCloud<FloatType> loadFromFile(const std::string& filename) {
		PointCloud<FloatType> pc;
		loadFromFile(filename, pc);
		return pc;
	}

	static void loadFromFile(const std::string& filename, PointCloud<FloatType>& pointCloud) {
		pointCloud.clear();
		std::string extension = util::getFileExtension(filename);

		if (extension == "ply") {
			loadFromPLY(filename, pointCloud);
		} else {
			throw MLIB_EXCEPTION("unknown file extension" + filename);
		}

		if (!pointCloud.isConsistent()) throw MLIB_EXCEPTION("inconsistent point cloud");
	}


	static void saveToFile(const std::string& filename, const std::vector<point3d<FloatType>> &points) {
		PointCloud<FloatType> pc;
		pc.m_points = points;
		saveToFile(filename, pc);
	}

	static void saveToFile(const std::string& filename, const PointCloud<FloatType>& pointCloud) {
		if (pointCloud.isEmpty()) {
			MLIB_WARNING("empty point cloud");
			return;
		}
		std::string extension = util::getFileExtension(filename);
		if (extension == "ply") {
			writeToPLY(filename, pointCloud);
		} else {
			throw MLIB_EXCEPTION("unknown file extension" + filename);
		}
	}


	/************************************************************************/
	/* Read Functions													    */
	/************************************************************************/

	static void loadFromPLY(const std::string& filename, PointCloud<FloatType>& pc);


	/************************************************************************/
	/* Write Functions													    */
	/************************************************************************/

	static void writeToPLY(const std::string& filename, const PointCloud<FloatType>& pc) {

		if (!std::is_same<FloatType, float>::value) throw MLIB_EXCEPTION("only implemented for float, not for double");

		std::ofstream file(filename, std::ios::binary);
		if (!file.is_open()) throw MLIB_EXCEPTION("Could not open file for writing " + filename);
		file << "ply\n";
		file << "format binary_little_endian 1.0\n";
		file << "comment MLIB generated\n";
		file << "element vertex " << pc.m_points.size() << "\n";
		file << "property float x\n";
		file << "property float y\n";
		file << "property float z\n";
		if (pc.m_normals.size() > 0) {
			file << "property float nx\n";
			file << "property float ny\n";
			file << "property float nz\n";
		}
		if (pc.m_colors.size() > 0) {
			file << "property uchar red\n";
			file << "property uchar green\n";
			file << "property uchar blue\n";
			file << "property uchar alpha\n";
		}
		file << "end_header\n";

		if (pc.m_colors.size() > 0 || pc.m_normals.size() > 0) {
			size_t vertexByteSize = sizeof(float)*3;
			if (pc.m_normals.size() > 0)	vertexByteSize += sizeof(float)*3;
			if (pc.m_colors.size() > 0)		vertexByteSize += sizeof(unsigned char)*4;
			BYTE* data = new BYTE[vertexByteSize*pc.m_points.size()];
			size_t byteOffset = 0;
			for (size_t i = 0; i < pc.m_points.size(); i++) {
				memcpy(&data[byteOffset], &pc.m_points[i], sizeof(float)*3);
				byteOffset += sizeof(float)*3;
				if (pc.m_normals.size() > 0) {
					memcpy(&data[byteOffset], &pc.m_normals[i], sizeof(float)*3);
					byteOffset += sizeof(float)*3;
				}
				if (pc.m_colors.size() > 0) {
					vec4uc c(pc.m_colors[i]*255);
					memcpy(&data[byteOffset], &c, sizeof(unsigned char)*4);
					byteOffset += sizeof(unsigned char)*4;
				}
			}
			file.write((const char*)data, byteOffset);
			SAFE_DELETE_ARRAY(data);
		} else {
			file.write((const char*)&pc.m_points[0], sizeof(float)*3*pc.m_points.size());
		}

		file.close();
	}


/*
	static void saveToFile(const std::string &filename, const std::vector<point3d<FloatType>> &points) {
		std::string extension = getFileExtension(filename);

		if (extension == "ply") {
			saveToPLYNoNormals(filename, points);
		} else if (extension == "pwn") {
			saveToPWN(filename, points);
		} else if (extension == "pcc") {
			saveToPCC(filename, points);
		} else {
			throw MLIB_EXCEPTION("unknown file format " + filename);
		}
	}

	static void saveToFile(const std::string &filename, const std::vector<point3d<FloatType>>* points, const std::vector<point3d<FloatType>>* normals = NULL, const std::vector<point3d<FloatType>>* colors = NULL) {
		std::string extension = getFileExtension(filename);

		assert(points);

		if (colors == NULL && normals == NULL) {
			saveToFile(filename, *points);	
		} else if (normals == NULL)	{
			if (extension == "ply") {
				saveToPLYColors(filename, *points, *colors);
			} else {
				throw MLIB_EXCEPTION("unknown file format " + filename);
			}
		} else {
			throw MLIB_EXCEPTION("unknown file format " + filename);
		}
	}

private:

	static void saveToPLYNoNormals(const std::string &filename, const std::vector<point3d<FloatType>> &points) {
		std::ofstream file(filename);
		if (!file.is_open())	throw std::ios::failure(__FUNCTION__ + std::string(": could not open file ") + filename);	

		//write header
		file << 
			"ply\n" <<
			"format ascii 1.0\n" <<
			"element vertex " << points.size() << "\n" <<
			"property float x\n" <<
			"property float y\n" <<
			"property float z\n" <<
			"end_header\n";

		for (unsigned int i = 0; i < points.size(); i++) {
			file << points[i].x << " " << points[i].y << " " << points[i].z << "\n";
		}
		
		file.close();
	}

	static void saveToPLYColors(const std::string &filename, const std::vector<point3d<FloatType>> &points, const std::vector<point3d<FloatType>> &colors) {
		std::ofstream file(filename);
		if (!file.is_open())	throw std::ios::failure(__FUNCTION__ + std::string(": could not open file ") + filename);

		//write header
		file << 
			"ply\n" <<
			"format ascii 1.0\n" <<
			"element vertex " << points.size() << "\n" <<
			"property float x\n" <<
			"property float y\n" <<
			"property float z\n" <<
			"property uchar red\n" <<
			"property uchar green\n" <<
			"property uchar blue\n" <<
			"end_header\n";

		for (unsigned int i = 0; i < points.size(); i++) {
			vec3ui c((unsigned int)(colors[i].x*(FloatType)255.0), (unsigned int)(colors[i].y*(FloatType)255.0), (unsigned int)(colors[i].z*(FloatType)255.0));
			math::clamp(c.x, (unsigned int)0, (unsigned int)255);
			math::clamp(c.y, (unsigned int)0, (unsigned int)255);
			math::clamp(c.z, (unsigned int)0, (unsigned int)255);
			file << points[i].x << " " << points[i].y << " " << points[i].z << " " << c.x << " " << c.y << " " << c.z << "\n";
		}
		
		file.close();
	}

	static void saveToPWN(const std::string &filename, const std::vector<point3d<FloatType>> &points) {
		std::ofstream file(filename);
		if (!file.is_open())	throw std::ios::failure(__FUNCTION__ + std::string(": could not open file ") + filename);	
			
		const unsigned int nVertices = (unsigned int)points.size();

		file << nVertices << std::endl;

		for(unsigned int i = 0; i < nVertices; i++)
		{
			file << points[i][0] << " ";
			file << points[i][1] << " ";
			file << points[i][2] << std::endl;
		}

		for(unsigned int i = 0; i < nVertices; i++)
		{
			file << "0.0" << " ";
			file << "0.0" << " ";
			file << "0.0" << std::endl;
		}
		
		file.close();
	}

	static void saveToPCC(const std::string &filename, const std::vector<point3d<FloatType>> &points) {
		std::ofstream file(filename);
		if (!file.is_open())	throw std::ios::failure(__FUNCTION__ + std::string(": could not open file ") + filename);	
			
		const unsigned int nVertices = (unsigned int)points.size();

		file << nVertices << std::endl;

		for(unsigned int i = 0; i < nVertices; i++)
		{
			file << points[i][0] << " ";
			file << points[i][1] << " ";
			file << points[i][2] << std::endl;
		}

		for(unsigned int i = 0; i < nVertices; i++)
		{
			float c = 255.0f*(i/(float)nVertices);

			file << (unsigned short)c << " ";
			file << (unsigned short)0 << " ";
			file << (unsigned short)0 << " ";
			file << (unsigned short)255 << std::endl;
		}
		
		file.close();
	}
*/

};

typedef PointCloudIO<float> PointCloudIOf;
typedef PointCloudIO<double> PointCloudIOd;

} //namespace ml


#include "pointCloudIO.cpp"

#endif
