#pragma once


//TODO CLEAN UP THIS CLASS!!!

#ifndef _POINT_CLOUD_IO_H_
#define _POINT_CLOUD_IO_H_

namespace ml {

template <class FloatType>
class PointCloudIO {
public:
	static void readFromFile(const std::string &filename, std::vector<point3d<FloatType>> &points) {
		std::string extension = getFileExtension(filename);

		if (extension == "ply") {
			ReadFromPLYNoNormals(filename, points);
		} else {
			throw MLIB_EXCEPTION("unknown file format " + filename);
		}
	}

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

		if (extension == "ply" && colors && normals) {
			saveToPLYColorsNormals(filename, *points, *normals, *colors);
		} 
		else if (colors == NULL && normals == NULL) {
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

	static void saveToPLYColorsNormals(const std::string &filename, const std::vector<point3d<FloatType>> &points, const std::vector<point3d<FloatType>> &normals, const std::vector<point3d<FloatType>> &colors) {
		std::ofstream file(filename);
		if (!file.is_open())	throw std::ios::failure(__FUNCTION__ + std::string(": could not open file ") + filename);

		if (points.size() != normals.size() ||
			points.size() != colors.size()) throw MLIB_EXCEPTION("invalid normal/color/point dimension");

		//write header
		file << 
			"ply\n" <<
			"format ascii 1.0\n" <<
			"element vertex " << points.size() << "\n" <<
			"property float x\n" <<
			"property float y\n" <<
			"property float z\n" <<
			"property float nx\n" <<
			"property float ny\n" <<
			"property float nz\n" <<
			"property uchar red\n" <<
			"property uchar green\n" <<
			"property uchar blue\n" <<
			"end_header\n";

		for (unsigned int i = 0; i < points.size(); i++) {
			vec3ui c((unsigned int)(colors[i].x*(FloatType)255.0), (unsigned int)(colors[i].y*(FloatType)255.0), (unsigned int)(colors[i].z*(FloatType)255.0));
			math::clamp(c.x, (unsigned int)0, (unsigned int)255);
			math::clamp(c.y, (unsigned int)0, (unsigned int)255);
			math::clamp(c.z, (unsigned int)0, (unsigned int)255);
			file 
				<< points[i].x << " " << points[i].y << " " << points[i].z << " " 
				<< normals[i].x << " " << normals[i].y << " " << normals[i].z << " "
				<< c.x << " " << c.y << " " << c.z << " " 
				<<"\n";
		}

		file.close();
	}

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

	struct PlyHeader {
		PlyHeader() {
			m_NumVertices = -1;
			m_NumFaces = -1;
		}
		unsigned int m_NumVertices;
		unsigned int m_NumFaces;
	};

	static void PlyHeaderLine(const std::string& line, PlyHeader& header) {

		std::stringstream ss(line);
		std::string currWord;
		ss >> currWord;

		if (currWord == "element") {
			ss >> currWord;
			if (currWord == "vertex") {
				ss >> header.m_NumVertices;
			} else if (currWord == "face") {
				ss >> header.m_NumFaces;
			}
		} 
		//else {
		//	std::cout << __FUNCTION__ << " ignoring header line: " << line << std::endl;
		//}
	}

	static void ReadFromPLYNoNormals(const std::string& filename, std::vector<point3d<FloatType>> &points) {
		std::ifstream file(filename);
		if (!file.is_open())	throw std::ios::failure(__FUNCTION__ + std::string(": could not open file ") + filename);			

		// read header
		PlyHeader header;

		std::string line;
		std::getline(file, line);
		int i1 = 0;
		while (line.compare(std::string("end_header"))) {
			PlyHeaderLine(line, header);
			std::getline(file, line);
		}

		assert(header.m_NumFaces == -1 || header.m_NumFaces == 0);	//make sure we got no faces for a point cloud
		assert(header.m_NumVertices != -1);

		points.resize(header.m_NumVertices);

		for (unsigned int i = 0; i < header.m_NumVertices; i++) {
			std::getline(file, line);
			std::stringstream ss(line);
			ss >> points[i].x >> points[i].y >> points[i].z;
		}
	}

	//! Returns the file extension in lower case
	static std::string getFileExtension(const std::string& filename) {
		std::string extension = filename.substr(filename.find_last_of(".")+1);
		for (unsigned int i = 0; i < extension.size(); i++) {
			extension[i] = (char)tolower(extension[i]);
		}
		return extension;
	}
};

typedef PointCloudIO<float> PointCloudIOf;
typedef PointCloudIO<double> PointCloudIOd;

} //namespace ml

#endif
