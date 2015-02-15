
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

		mesh.deleteRedundantIndices();

		if (!mesh.isConsistent()) {
			throw MLIB_EXCEPTION("inconsistent mesh data: " + filename);
		}
	}

	static void saveToFile(const std::string& filename, const MeshData<FloatType>& mesh) {

		if (mesh.isEmpty()) {		
			MLIB_WARNING("empty mesh");
			return;
		}

		if (!mesh.isConsistent()) {
			throw MLIB_EXCEPTION("inconsistent mesh data: " + filename);
		}

		std::string extension = util::getFileExtension(filename);

		if (extension == "off") {
			saveToOFF(filename, mesh);
		} else if (extension == "ply") {
			saveToPLY(filename, mesh);
		} else if (extension == "obj") {
			saveToOBJ(filename, mesh);
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

	static void saveToPLY(const std::string& filename, const MeshData<FloatType>& mesh);

	static void saveToOFF(const std::string& filename, const MeshData<FloatType>& mesh);

	static void saveToOBJ(const std::string& filename, const MeshData<FloatType>& mesh);

private:

#define OBJ_LINE_BUF_SIZE 256
	static void skipLine(char * buf, int size, FILE * fp)
	{
		//some weird files don't have newlines, which confused fgets
		while (1) {
			int c = fgetc(fp);
			if (c == EOF || c == '\n' || c == '\r') break;
		}
		//do {
		//	buf[size-1] = '$';
		//	fgets(buf, size, fp);
		//} while (buf[size-1] != '$');
	}
};

typedef MeshIO<float>	MeshIOf;
typedef MeshIO<double>	MeshIOd;

}  // namespace ml

#include "meshIO.cpp"

#endif  // CORE_MESH_MESHIO_H_