
#ifndef CORE_MESH_MESHIO_H_
#define CORE_MESH_MESHIO_H_

namespace ml {

template <class FloatType>
class MeshIO {

public:

	static MeshData<FloatType> loadFromFile(const std::string& filename) {
		MeshData<FloatType> data;	
		loadFromFile(filename, data);
		return data;
	}

	static void loadFromFile(const std::string& filename, MeshData<FloatType>& mesh) {
		mesh.clear();
		std::string extension = util::getFileExtension(filename);

		if (extension == "off") {
			loadFromOFF(filename, mesh);
		} else if (extension == "ply") {
			loadFromPLY(filename, mesh);
		} else if (extension == "obj") {
			loadFromOBJ(filename, mesh);
		} else 	{
			throw MLIB_EXCEPTION("unknown file format: " + filename);
		}

		if (!mesh.isConsistent()) {
			throw MLIB_EXCEPTION("inconsistent mesh data: " + filename);
		}
	}

	static void writeToFile(const std::string& filename, const MeshData<FloatType>& mesh) {

		if (!mesh.isConsistent()) {
			throw MLIB_EXCEPTION("inconsistent mesh data: " + filename);
		}

		std::string extension = util::getFileExtension(filename);

		if (extension == "off") {
			writeToOFF(filename, mesh);
		} else if (extension == "ply") {
			writeToPLY(filename, mesh);
		} else if (extension == "obj") {
			writeToOBJ(filename, mesh);
		} else {
			throw MLIB_EXCEPTION("unknown file format: " + filename);
		}
	}


	/************************************************************************/
	/* Read Functions													    */
	/************************************************************************/

	static void loadFromPLY(const std::string& filename, MeshData<FloatType>& mesh);

	static void loadFromOFF(const std::string& filename, MeshData<FloatType>& mesh);

	static void loadFromOBJ(const std::string& filename, MeshData<FloatType>& mesh);


	/************************************************************************/
	/* Write Functions													    */
	/************************************************************************/

	static void writeToPLY(const std::string& filename, const MeshData<FloatType>& mesh);

	static void writeToOFF(const std::string& filename, const MeshData<FloatType>& mesh);

	static void writeToOBJ(const std::string& filename, const MeshData<FloatType>& mesh);

private:

#define OBJ_LINE_BUF_SIZE 256
	static void skipLine(char * buf, int size, FILE * fp)
	{
		do {
			buf[size-1] = '$';
			fgets(buf, size, fp);
		} while (buf[size-1] != '$');
	}

	struct PlyHeader {
		struct PlyProperty {
			PlyProperty() {
				byteSize = 0;
			}
			std::string name;
			unsigned int byteSize;
		};
		PlyHeader() {
			m_NumVertices = -1;
			m_NumFaces = -1;
			m_bHasNormals = false;
			m_bHasColors = false;
		}
		unsigned int m_NumVertices;
		unsigned int m_NumFaces;
		std::vector<PlyProperty> m_Properties;
		bool m_bBinary;
		bool m_bHasNormals;
		bool m_bHasColors;
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
		else if(currWord == "format") {
			ss >> currWord;
			if (currWord == "binary_little_endian")	{
				header.m_bBinary = true;
			} else {
				header.m_bBinary = false;
			}
		}
		else if (currWord == "property") {
			if (!util::endsWith(line, "vertex_indices")) {
				PlyHeader::PlyProperty p;
				std::string which;
				ss >> which;
				ss >> p.name;
				if (p.name == "nx")	header.m_bHasNormals = true;
				if (p.name == "red") header.m_bHasColors = true;
				if (which == "float") p.byteSize = 4;
				if (which == "uchar" || which == "char") p.byteSize = 1;
				header.m_Properties.push_back(p);
			}
		}
	}
};

typedef MeshIO<float>	MeshIOf;
typedef MeshIO<double>	MeshIOd;

}  // namespace ml

#include "meshIO.cpp"

#endif  // CORE_MESH_MESHIO_H_