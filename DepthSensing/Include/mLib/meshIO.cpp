
#ifndef CORE_MESH_MESHIO_INL_H_
#define CORE_MESH_MESHIO_INL_H_

namespace ml {

template <class FloatType>
void MeshIO<FloatType>::loadFromPLY( const std::string& filename, MeshData<FloatType>& mesh )
{
	mesh.clear();
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open())	throw MLIB_EXCEPTION("Could not open file " + filename);			

	// read header
	PlyHeader header;

	std::string line;
	std::getline(file, line);
	while (line.find("end_header") == std::string::npos) {
		PlyHeaderLine(line, header);
		std::getline(file, line);
	}

	assert(header.m_NumFaces != -1);
	assert(header.m_NumVertices != -1);

	mesh.m_Vertices.resize(header.m_NumVertices);
	mesh.m_FaceIndicesVertices.resize(header.m_NumFaces);
	if (header.m_bHasNormals) mesh.m_Normals.resize(header.m_NumVertices);
	if (header.m_bHasColors) mesh.m_Colors.resize(header.m_NumVertices);

	if(header.m_bBinary)
	{
		//unsigned int size = 3*4+3*4+3+11*4;
		unsigned int size = 0;
		for (unsigned int i = 0; i < header.m_Properties.size(); i++) {
			size += header.m_Properties[i].byteSize;
		}
		//size = 3*4+3*4+3+11*4;
		char* data = new char[size*header.m_NumVertices];
		file.read(data, size*header.m_NumVertices);
		for (unsigned int i = 0; i < header.m_NumVertices; i++) {
			unsigned int byteOffset = 0;
			for (unsigned int j = 0; j < header.m_Properties.size(); j++) {
				if (header.m_Properties[j].name == "x") {
					mesh.m_Vertices[i].x = ((float*)&data[i*size + byteOffset])[0];
					byteOffset += header.m_Properties[j].byteSize;
				}
				else if (header.m_Properties[j].name == "y") {
					mesh.m_Vertices[i].y = ((float*)&data[i*size + byteOffset])[0];
					byteOffset += header.m_Properties[j].byteSize;
				}
				else if (header.m_Properties[j].name == "z") {
					mesh.m_Vertices[i].z = ((float*)&data[i*size + byteOffset])[0];
					byteOffset += header.m_Properties[j].byteSize;
				}
				else if (header.m_Properties[j].name == "nx") {
					mesh.m_Normals[i].x = ((float*)&data[i*size + byteOffset])[0];
					byteOffset += header.m_Properties[j].byteSize;
				}
				else if (header.m_Properties[j].name == "ny") {
					mesh.m_Normals[i].y = ((float*)&data[i*size + byteOffset])[0];
					byteOffset += header.m_Properties[j].byteSize;
				}
				else if (header.m_Properties[j].name == "nz") {
					mesh.m_Normals[i].z = ((float*)&data[i*size + byteOffset])[0];
					byteOffset += header.m_Properties[j].byteSize;
				}
				else if (header.m_Properties[j].name == "red") {
					mesh.m_Colors[i].x = ((unsigned char*)&data[i*size + byteOffset])[0];	mesh.m_Colors[i].x/=255.0f;
					byteOffset += header.m_Properties[j].byteSize;
				}
				else if (header.m_Properties[j].name == "green") {
					mesh.m_Colors[i].y = ((unsigned char*)&data[i*size + byteOffset])[0];	mesh.m_Colors[i].y/=255.0f;
					byteOffset += header.m_Properties[j].byteSize;
				}
				else if (header.m_Properties[j].name == "blue") {
					mesh.m_Colors[i].z = ((unsigned char*)&data[i*size + byteOffset])[0];	mesh.m_Colors[i].z/=255.0f;
					byteOffset += header.m_Properties[j].byteSize;
				}
				else if (header.m_Properties[j].name == "alpha") {
					mesh.m_Colors[i].w = ((unsigned char*)&data[i*size + byteOffset])[0];	mesh.m_Colors[i].w/=255.0f;
					byteOffset += header.m_Properties[j].byteSize;
				}
			}
			assert(byteOffset == size);

		}	

		delete [] data;

		size = 1+3*4;	//typically 1 uchar for numVertices per triangle, 3 * int for indeices
		data = new char[size*header.m_NumFaces];
		file.read(data, size*header.m_NumFaces);
		for (unsigned int i = 0; i < header.m_NumFaces; i++) {	
			mesh.m_FaceIndicesVertices[i].push_back(((int*)&data[i*size+1])[0]);
			mesh.m_FaceIndicesVertices[i].push_back(((int*)&data[i*size+1])[1]);
			mesh.m_FaceIndicesVertices[i].push_back(((int*)&data[i*size+1])[2]);
		}

		//if (mesh.m_Colors.size() == 0) {
		//	mesh.m_Colors.resize(header.m_NumVertices);
		//	for (size_t i = 0; i < mesh.m_Colors.size(); i++) {
		//		mesh.m_Colors[i] = vec3f(0.5f, 0.5f, 0.5f);
		//	}
		//}
		delete [] data;

	}
	else
	{
		for (unsigned int i = 0; i < header.m_NumVertices; i++) {
			std::getline(file, line);
			std::stringstream ss(line);
			ss >> mesh.m_Vertices[i].x >> mesh.m_Vertices[i].y >> mesh.m_Vertices[i].z;
			if (header.m_bHasColors) {
				ss >> mesh.m_Colors[i].x >> mesh.m_Colors[i].y >> mesh.m_Colors[i].z;
				mesh.m_Colors[i] /= (FloatType)255.0;
			}
		}

		for (unsigned int i = 0; i < header.m_NumFaces; i++) {
			std::getline(file, line);
			std::stringstream ss(line);
			unsigned int num_vs;
			ss >> num_vs;
			for (unsigned int j = 0; j < num_vs; j++) {
				unsigned int idx;
				ss >> idx;
				mesh.m_FaceIndicesVertices[i].push_back(idx);
			}
		}

		//if (mesh.m_Colors.size() == 0) {
		//	mesh.m_Colors.resize(header.m_NumVertices);
		//	for (size_t i = 0; i < mesh.m_Colors.size(); i++) {
		//		mesh.m_Colors[i] = vec3f(0.5f, 0.5f, 0.5f);
		//	}
		//}

	}
}

template <class FloatType>
void MeshIO<FloatType>::loadFromOFF( const std::string& filename, MeshData<FloatType>& mesh )
{
	mesh.clear();

	std::ifstream file(filename);
	if (!file.is_open())	throw MLIB_EXCEPTION("Could not open file " + filename);			

	// first line should say 'OFF'
	char string1[5];
	file >> string1;

	// read header
	unsigned int numV = 0;
	unsigned int numP = 0;
	unsigned int numE = 0;
	file >> numV >> numP >> numE;

	mesh.m_Vertices.resize(numV);
	mesh.m_FaceIndicesVertices.resize(numP);
	mesh.m_Colors.resize(numV);

	if(std::string(string1).compare("OFF") == 0)
	{
		// read points
		for(unsigned int i = 0; i < numV; i++) {
			point3d<FloatType> v;
			file >> v.x >> v.y >> v.z;
			mesh.m_Vertices[i] = v;
			mesh.m_Colors[i] = vec3f::origin;
		}
	}
	else
	{	// ignore color
		// read points
		for (unsigned int i = 0; i < numV; i++) {
			point3d<FloatType> v;
			point4d<FloatType> c;
			file >> v.x >> v.y >> v.z;
			file >> c.x >> c.y >> c.z >> c.w;
			mesh.m_Vertices[i] = v;
			mesh.m_Colors[i] = c / 255;	//typically colors are stored in RGB \in [0;255]
		}
	}

	// read faces (i.e., indices)
	for(unsigned int i = 0; i < numP; i++) {
		unsigned int num_vs;
		file >> num_vs;
		for (unsigned int j = 0; j < num_vs; j++) {
			unsigned int idx;
			file >> idx;
			mesh.m_FaceIndicesVertices[i].push_back(idx);
		}
	}
}

template <class FloatType>
void MeshIO<FloatType>::loadFromOBJ( const std::string& filename, MeshData<FloatType>& mesh )
{
	mesh.clear();

	FILE *fp = NULL;
	//fopen_s(&fp, filename.c_str(), "r");
	fp = fopen(filename.c_str(), "r");
	if (!fp) throw MLIB_EXCEPTION("Could not open file " + filename);

	char buf[OBJ_LINE_BUF_SIZE];
	float val[6];
	int idx[256][3];
	int match;

	unsigned int type = 0;


	while ( fscanf( fp, "%s", buf) != EOF ) {

		switch (buf[0]) {
		case '#':
			//comment line, eat the remainder
			skipLine( buf, OBJ_LINE_BUF_SIZE, fp);
			break;
		case 'v':
			switch (buf[1]) {

			case '\0':
				//vertex, 3 or 4 components
				val[3] = 1.0f;  //default w coordinate
				match = fscanf( fp, "%f %f %f %f %f %f", &val[0], &val[1], &val[2], &val[3], &val[4], &val[5]);	//meshlab stores colors right after vertex pos (3 xyz, 3 rgb)
				mesh.m_Vertices.push_back(point3d<FloatType>(val[0], val[1], val[2]));

				if (match == 6) {  //we found color data
					mesh.m_Colors.push_back(point4d<FloatType>(val[3], val[4], val[5], (FloatType)1.0));
				}
				assert( match == 3 || match == 4 || match == 6);
				break;

			case 'n':
				//normal, 3 components
				match = fscanf( fp, "%f %f %f", &val[0], &val[1], &val[2]);
				mesh.m_Normals.push_back(point3d<FloatType>(val[0], val[1], val[2]));

				assert( match == 3);
				break;

			case 't':
				//texcoord, 2 or 3 components
				val[2] = 0.0f;  //default r coordinate
				match = fscanf( fp, "%f %f %f %f", &val[0], &val[1], &val[2], &val[3]);
				mesh.m_TextureCoords.push_back(point2d<FloatType>(val[0], val[1]));

				assert( match > 1 && match < 4);
				assert( match == 2);
				break;
			}
			break;

		case 'f':
			//face
			fscanf( fp, "%s", buf);
			{
				unsigned int n = 2;

				//determine the type, and read the initial vertex, all entries in a face must have the same format
				if ( sscanf( buf, "%d//%d", &idx[0][0], &idx[0][1]) == 2) {
					type = 4;
					//This face has vertex and normal indices

					//remap them to the right spot
					idx[0][0] = (idx[0][0] > 0) ? (idx[0][0] - 1) : ((int)mesh.m_Vertices.size() - idx[0][0]);
					idx[0][1] = (idx[0][1] > 0) ? (idx[0][1] - 1) : ((int)mesh.m_Normals.size() - idx[0][1]);

					//grab the second vertex to prime
					fscanf( fp, "%d//%d", &idx[1][0], &idx[1][1]);

					//remap them to the right spot
					idx[1][0] = (idx[1][0] > 0) ? (idx[1][0] - 1) : ((int)mesh.m_Vertices.size() - idx[1][0]);
					idx[1][1] = (idx[1][1] > 0) ? (idx[1][1] - 1) : ((int)mesh.m_Normals.size() - idx[1][1]);

					//create the fan
					while ( fscanf( fp, "%d//%d", &idx[n][0], &idx[n][1]) == 2) {
						//remap them to the right spot
						idx[n][0] = (idx[n][0] > 0) ? (idx[n][0] - 1) : ((int)mesh.m_Vertices.size() - idx[n][0]);
						idx[n][1] = (idx[n][1] > 0) ? (idx[n][1] - 1) : ((int)mesh.m_Normals.size() - idx[n][1]);
						n++;
					}
				}
				else if ( sscanf( buf, "%d/%d/%d", &idx[0][0], &idx[0][1], &idx[0][2]) == 3) {
					type = 3;
					//This face has vertex, texture coordinate, and normal indices

					//remap them to the right spot
					idx[0][0] = (idx[0][0] > 0) ? (idx[0][0] - 1) : ((int)mesh.m_Vertices.size() - idx[0][0]);
					idx[0][1] = (idx[0][1] > 0) ? (idx[0][1] - 1) : ((int)mesh.m_TextureCoords.size() - idx[0][1]);
					idx[0][2] = (idx[0][2] > 0) ? (idx[0][2] - 1) : ((int)mesh.m_Normals.size() - idx[0][2]);

					//grab the second vertex to prime
					fscanf( fp, "%d/%d/%d", &idx[1][0], &idx[1][1], &idx[1][2]);

					//remap them to the right spot
					idx[1][0] = (idx[1][0] > 0) ? (idx[1][0] - 1) : ((int)mesh.m_Vertices.size() - idx[1][0]);
					idx[1][1] = (idx[1][1] > 0) ? (idx[1][1] - 1) : ((int)mesh.m_TextureCoords.size() - idx[1][1]);
					idx[1][2] = (idx[1][2] > 0) ? (idx[1][2] - 1) : ((int)mesh.m_Normals.size() - idx[1][2]);

					//create the fan
					while ( fscanf( fp, "%d/%d/%d", &idx[n][0], &idx[n][1], &idx[n][2]) == 3) {
						//remap them to the right spot
						idx[n][0] = (idx[n][0] > 0) ? (idx[n][0] - 1) : ((int)mesh.m_Vertices.size() - idx[n][0]);
						idx[n][1] = (idx[n][1] > 0) ? (idx[n][1] - 1) : ((int)mesh.m_TextureCoords.size() - idx[n][1]);
						idx[n][2] = (idx[n][2] > 0) ? (idx[n][2] - 1) : ((int)mesh.m_Normals.size() - idx[n][2]);
						n++;
					}
				}
				else if ( sscanf( buf, "%d/%d/", &idx[0][0], &idx[0][1]) == 2) {
					type = 2;
					//This face has vertex and texture coordinate indices

					//remap them to the right spot
					idx[0][0] = (idx[0][0] > 0) ? (idx[0][0] - 1) : ((int)mesh.m_Vertices.size() - idx[0][0]);
					idx[0][1] = (idx[0][1] > 0) ? (idx[0][1] - 1) : ((int)mesh.m_TextureCoords.size() - idx[0][1]);

					//grab the second vertex to prime
					fscanf( fp, "%d/%d/", &idx[1][0], &idx[1][1]);

					//remap them to the right spot
					idx[1][0] = (idx[1][0] > 0) ? (idx[1][0] - 1) : ((int)mesh.m_Vertices.size() - idx[1][0]);
					idx[1][1] = (idx[1][1] > 0) ? (idx[1][1] - 1) : ((int)mesh.m_TextureCoords.size() - idx[1][1]);

					//create the fan
					while ( fscanf( fp, "%d/%d/", &idx[n][0], &idx[n][1]) == 2) {
						//remap them to the right spot
						idx[n][0] = (idx[n][0] > 0) ? (idx[n][0] - 1) : ((int)mesh.m_Vertices.size() - idx[n][0]);
						idx[n][1] = (idx[n][1] > 0) ? (idx[n][1] - 1) : ((int)mesh.m_TextureCoords.size() - idx[n][1]);
						n++;
					}
				}
				else if ( sscanf( buf, "%d/%d", &idx[0][0], &idx[0][1]) == 2) {
					type = 2;
					//This face has vertex and texture coordinate indices

					//remap them to the right spot
					idx[0][0] = (idx[0][0] > 0) ? (idx[0][0] - 1) : ((int)mesh.m_Vertices.size() - idx[0][0]);
					idx[0][1] = (idx[0][1] > 0) ? (idx[0][1] - 1) : ((int)mesh.m_TextureCoords.size() - idx[0][1]);

					//grab the second vertex to prime
					fscanf( fp, "%d/%d", &idx[1][0], &idx[1][1]);

					//remap them to the right spot
					idx[1][0] = (idx[1][0] > 0) ? (idx[1][0] - 1) : ((int)mesh.m_Vertices.size() - idx[1][0]);
					idx[1][1] = (idx[1][1] > 0) ? (idx[1][1] - 1) : ((int)mesh.m_TextureCoords.size() - idx[1][1]);

					//create the fan
					while ( fscanf( fp, "%d/%d", &idx[n][0], &idx[n][1]) == 2) {
						//remap them to the right spot
						idx[n][0] = (idx[n][0] > 0) ? (idx[n][0] - 1) : ((int)mesh.m_Vertices.size() - idx[n][0]);
						idx[n][1] = (idx[n][1] > 0) ? (idx[n][1] - 1) : ((int)mesh.m_TextureCoords.size() - idx[n][1]);
						n++;
					}
				}
				else if ( sscanf( buf, "%d", &idx[0][0]) == 1) {
					type = 1;
					//This face has only vertex indices

					//remap them to the right spot
					idx[0][0] = (idx[0][0] > 0) ? (idx[0][0] - 1) : ((int)mesh.m_Vertices.size() - idx[0][0]);

					//grab the second vertex to prime
					fscanf( fp, "%d", &idx[1][0]);

					//remap them to the right spot
					idx[1][0] = (idx[1][0] > 0) ? (idx[1][0] - 1) : ((int)mesh.m_Vertices.size() - idx[1][0]);

					//create the fan
					while ( fscanf( fp, "%d", &idx[n][0]) == 1) {
						//remap them to the right spot
						idx[n][0] = (idx[n][0] > 0) ? (idx[n][0] - 1) : ((int)mesh.m_Vertices.size() - idx[n][0]);
						n++;
					}
				}
				else {
					throw MLIB_EXCEPTION(filename + ": broken obj (face line invalid)");
					//skipLine( buf, OBJ_LINE_BUF_SIZE, fp);
				}

				if (n < 3)	throw MLIB_EXCEPTION(filename + ": broken obj (face with less than 3 indices)");

				//create face
				std::vector<unsigned int> currFaceIndicesVertices;
				std::vector<unsigned int> currFaceIndicesNormals;
				std::vector<unsigned int> currFaceIndicesTextureCoords;

				for (unsigned int i = 0; i < n; i++) {	
					currFaceIndicesVertices.push_back(idx[i][0]);
					if (type == 2) {	//has vertex and tex coords
						currFaceIndicesTextureCoords.push_back(idx[i][1]);
					} else if (type == 3) {
						currFaceIndicesTextureCoords.push_back(idx[i][1]);
						currFaceIndicesNormals.push_back(idx[i][2]);
					}
					//vertex[i] = vertices[idx[i][0]];
					//	if (type == 2) {	//has vertex and tex coords
					//		vertex[i]->texCoord = float2(_texCoords[idx[i][1]*3+0], _texCoords[idx[i][1]*3+1]);
					//	} else if (type == 3) { // has vertex, normals and tex coords
					//		vertex[i]->texCoord = float2(_texCoords[idx[i][2]*3+0], _texCoords[idx[i][2]*3+1]);
					//	}
				}

				if (currFaceIndicesVertices.size())			mesh.m_FaceIndicesVertices.push_back(currFaceIndicesVertices);
				if (currFaceIndicesNormals.size())			mesh.m_FaceIndicesNormals.push_back(currFaceIndicesNormals);
				if (currFaceIndicesTextureCoords.size())	mesh.m_FaceIndicesTextureCoords.push_back(currFaceIndicesTextureCoords);
			}

			break;

		case 's':
		case 'g':
		case 'u':
			//all presently ignored
		default:
			skipLine( buf, OBJ_LINE_BUF_SIZE, fp);

		};
	}

	fclose(fp);
}




template <class FloatType>
void MeshIO<FloatType>::writeToPLY( const std::string& filename, const MeshData<FloatType>& mesh )
{
	if (!std::is_same<FloatType, float>::value) throw MLIB_EXCEPTION("only implemented for float not for double");

	std::ofstream file(filename, std::ios::binary);
	if (!file.is_open()) throw MLIB_EXCEPTION("Could not open file for writing " + filename);
	file << "ply\n";
	file << "format binary_little_endian 1.0\n";
	file << "comment MLIB generated\n";
	file << "element vertex " << mesh.m_Vertices.size() << "\n";
	file << "property float x\n";
	file << "property float y\n";
	file << "property float z\n";
	if (mesh.m_Normals.size() > 0) {
		file << "property float nx\n";
		file << "property float ny\n";
		file << "property float nz\n";
	}
	if (mesh.m_Colors.size() > 0) {
		file << "property uchar red\n";
		file << "property uchar green\n";
		file << "property uchar blue\n";
		file << "property uchar alpha\n";
	}
	file << "element face " << mesh.m_FaceIndicesVertices.size() << "\n";
	file << "property list uchar int vertex_indices\n";
	file << "end_header\n";

	//TODO make this more efficient: i.e., copy first into an array, and then perform just a single write
	if (mesh.m_Colors.size() > 0 || mesh.m_Normals.size() > 0) {
		//for (size_t i = 0; i < mesh.m_Vertices.size(); i++) {
		//	file.write((const char*)&mesh.m_Vertices[i], sizeof(float)*3);
		//	if (mesh.m_Normals.size() > 0) {
		//		file.write((const char*)&mesh.m_Normals[i], sizeof(float)*3);
		//	}
		//	if (mesh.m_Colors.size() > 0) {
		//		vec4uc c(mesh.m_Colors[i]*255.0f);
		//		file.write((const char*)&c, sizeof(unsigned char)*4);
		//	}
		//}

		size_t vertexByteSize = sizeof(float)*3;
		if (mesh.m_Normals.size() > 0)	vertexByteSize += sizeof(float)*3;
		if (mesh.m_Colors.size() > 0)	vertexByteSize += sizeof(unsigned char)*4;
		BYTE* data = new BYTE[vertexByteSize*mesh.m_Vertices.size()];
		size_t byteOffset = 0;
		for (size_t i = 0; i < mesh.m_Vertices.size(); i++) {
			memcpy(&data[byteOffset], &mesh.m_Vertices[i], sizeof(float)*3);
			byteOffset += sizeof(float)*3;
			if (mesh.m_Normals.size() > 0) {
				memcpy(&data[byteOffset], &mesh.m_Normals[i], sizeof(float)*3);
				byteOffset += sizeof(float)*3;
			}
			if (mesh.m_Colors.size() > 0) {
				vec4uc c(mesh.m_Colors[i]*255);
				memcpy(&data[byteOffset], &c, sizeof(unsigned char)*4);
				byteOffset += sizeof(unsigned char)*4;
			}
		}
		file.write((const char*)data, byteOffset);
		SAFE_DELETE_ARRAY(data);
	} else {
		file.write((const char*)&mesh.m_Vertices[0], sizeof(float)*3*mesh.m_Vertices.size());
	}
	for (size_t i = 0; i < mesh.m_FaceIndicesVertices.size(); i++) {
		unsigned char numFaceIndices = (unsigned char)mesh.m_FaceIndicesVertices[i].size();
		file.write((const char*)&numFaceIndices, sizeof(unsigned char));
		file.write((const char*)&mesh.m_FaceIndicesVertices[i][0], numFaceIndices*sizeof(unsigned int));
	}
	file.close();
}


template <class FloatType>
void MeshIO<FloatType>::writeToOFF( const std::string& filename, const MeshData<FloatType>& mesh )
{
	std::ofstream file(filename);
	if (!file.is_open())	throw MLIB_EXCEPTION("Could not open file for writing " + filename);		

	// first line should say 'COFF'
	file << "COFF\n";

	// write header (verts, faces, edges)
	file << mesh.m_Vertices.size() << " " << mesh.m_FaceIndicesVertices.size() << " " << 0 << "\n";

	// write points
	for (size_t i = 0; i < mesh.m_Vertices.size(); i++) {
		file << mesh.m_Vertices[i].x << " " << mesh.m_Vertices[i].y << " " << mesh.m_Vertices[i].z;
		if (mesh.m_Colors.size() > 0) {
			file << " " << 
				(unsigned int)(mesh.m_Colors[i].x*255) << " " << 
				(unsigned int)(mesh.m_Colors[i].y*255) << " " << 
				(unsigned int)(mesh.m_Colors[i].z*255) << " " << 
				(unsigned int)(mesh.m_Colors[i].w*255) << " ";
		}
		file << "\n";
	}

	// write faces
	for (size_t i = 0; i < mesh.m_FaceIndicesVertices.size(); i++) {
		file << mesh.m_FaceIndicesVertices[i].size();
		for (size_t j = 0; j < mesh.m_FaceIndicesVertices[i].size(); j++) {
			file << " " << mesh.m_FaceIndicesVertices[i][j];
		}
		file << "\n";
	}

	file.close();
}


template <class FloatType>
void MeshIO<FloatType>::writeToOBJ( const std::string& filename, const MeshData<FloatType>& mesh )
{
	std::ofstream file(filename);
	if (!file.is_open())	throw MLIB_EXCEPTION("Could not open file for writing " + filename);

	file << "####\n";
	file << "#\n";
	file << "# OBJ file Generated by MLIB\n";
	file << "#\n";
	file << "####\n";
	file << "# Object " << filename << "\n";
	file << "#\n";
	file << "# Vertices: " << mesh.m_Vertices.size() << "\n";
	file << "# Faces: " << mesh.m_FaceIndicesVertices.size() << "\n";
	file << "#\n";
	file << "####\n";

	for (size_t i = 0; i < mesh.m_Vertices.size(); i++) {
		file << "v ";
		file << mesh.m_Vertices[i].x << " " << mesh.m_Vertices[i].y << " " << mesh.m_Vertices[i].z;
		if (mesh.m_Colors.size() > 0) {
			file << " " << mesh.m_Colors[i].x << " " << mesh.m_Colors[i].y << " " << mesh.m_Colors[i].z;
		}
		file << "\n";
	}
	for (size_t i = 0; i < mesh.m_Normals.size(); i++) {
		file << "vn ";
		file << mesh.m_Normals[i].x << " " << mesh.m_Normals[i].y << " " << mesh.m_Normals[i].z << "\n";
	}
	for (size_t i = 0; i < mesh.m_TextureCoords.size(); i++) {
		file << "vt ";
		file << mesh.m_TextureCoords[i].x << " " << mesh.m_TextureCoords[i].y << "\n";
	}
	for (size_t i = 0; i < mesh.m_FaceIndicesVertices.size(); i++) {
		file << "f ";
		for (size_t j = 0; j < mesh.m_FaceIndicesVertices[i].size(); j++) {
			file << mesh.m_FaceIndicesVertices[i][j]+1;
			if (mesh.m_FaceIndicesTextureCoords.size() > 0 || mesh.m_FaceIndicesNormals.size() > 0) {
				file << "//";
				if (mesh.m_FaceIndicesTextureCoords.size() > 0) {
					file << mesh.m_FaceIndicesTextureCoords[i][j]+1;
				}
				file << "//";
				if (mesh.m_FaceIndicesNormals.size() > 0) {
					file << mesh.m_FaceIndicesNormals[i][j]+1;
				}
			}
			file << " ";
		}
		file << "\n";
	}

	file.close();
}

}  // namespace ml

#endif  // CORE_MESH_MESHIO_INL_H_