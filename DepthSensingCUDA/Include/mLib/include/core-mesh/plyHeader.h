
#ifndef CORE_MESH_PLYHEADER_H_
#define CORE_MESH_PLYHEADER_H_

namespace ml {

	struct PlyHeader {
		struct PlyProperty {
			PlyProperty() {
				byteSize = 0;
			}
			std::string name;
			unsigned int byteSize;
		};
		PlyHeader(std::ifstream& file) {
			m_NumVertices = (unsigned int)-1;
			m_NumFaces = (unsigned int)-1;
			m_bHasNormals = false;
			m_bHasColors = false;

			read(file);
		}
		PlyHeader() {
			m_NumVertices = (unsigned int)-1;
			m_NumFaces = (unsigned int)-1;
			m_bHasNormals = false;
			m_bHasColors = false;
		}
		unsigned int m_NumVertices;
		unsigned int m_NumFaces;
		std::vector<PlyProperty> m_Properties;
		bool m_bBinary;
		bool m_bHasNormals;
		bool m_bHasColors;

		void read(std::ifstream& file) {
			std::string line;
			std::getline(file, line);
			while (line.find("end_header") == std::string::npos) {
				PlyHeaderLine(line, *this);
				std::getline(file, line);
			}
		}

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


} // namespace ml

#endif