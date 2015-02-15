#ifndef EXT_OPENMESH_TRIMESH_H_
#define EXT_OPENMESH_TRIMESH_H_

#include <string>

namespace ml {
	//////////////////////////////////////////////////////////////////////////
	// TriMesh Operations (simplification, fairing etc)						//
	//////////////////////////////////////////////////////////////////////////

	namespace OpenMeshTriMesh {


		//struct Traits : OpenMesh::DefaultTraits
		//{
		//	typedef OpenMesh::Vec3f Point;
		//	typedef OpenMesh::Vec3f Normal; 
		//	typedef OpenMesh::Vec4f Color;
		//	typedef OpenMesh::Vec2f TexCoord2D;

		//	//VertexAttributes(OpenMesh::Attributes::Normal | OpenMesh::Attributes::Color | OpenMesh::Attributes::TexCoord2D );
		//	//FaceAttributes(Attributes::Status | Attributes::Normal);
		//	//EdgeAttributes(Attributes::Status);
		//};

		//struct Traits : OpenMesh::DefaultTraits
		//{
		//	typedef OpenMesh::Vec3f Point;
		//	typedef OpenMesh::Vec3f Normal; 
		//	typedef OpenMesh::Vec3uc Color;
		//	typedef float Scalar;


		//	VertexAttributes(OpenMesh::Attributes::Status| OpenMesh::Attributes::Normal | OpenMesh::Attributes::Color );
		//	FaceAttributes(OpenMesh::Attributes::Status | OpenMesh::Attributes::Normal);
		//	EdgeAttributes(OpenMesh::Attributes::Status);
		//};

		typedef OpenMesh::TriMesh_ArrayKernelT<> Mesh;					//mesh type
		typedef OpenMesh::Decimater::DecimaterT< Mesh > Decimater;				// Decimater type
		typedef OpenMesh::Decimater::ModQuadricT< Mesh >::Handle HModQuadric;	// Decimation Module Handle type


		/*
		static bool Vec3fless(const OpenMesh::Vec3f& v0, const OpenMesh::Vec3f& v1)
		{
			if (v0[0] < v1[0]) return true;
			if (v0[0] > v1[0]) return false;
			if (v0[1] < v1[1]) return true;
			if (v0[1] > v1[1]) return false;
			if (v0[2] < v1[2]) return true;

			return false;
		}


		unsigned int removeDuplicateVertices(std::vector<OpenMesh::Vec3f>& verts, std::vector<unsigned int>& tris)
		{
			int numV = (int)verts.size();
			int numT = (int)tris.size();

			std::map<OpenMesh::Vec3f, int, bool(*)(const OpenMesh::Vec3f&, const OpenMesh::Vec3f&)> pts(Vec3fless);

			std::vector<unsigned int> vertexLookUp;
			vertexLookUp.resize(numV);
			std::vector<OpenMesh::Vec3f> new_verts;

			int cnt = 0;
			for(int i1 = 0; i1 < numV; i1++)
			{
				OpenMesh::Vec3f pt = verts[i1];

				std::map<OpenMesh::Vec3f, int, bool(*)(const OpenMesh::Vec3f&, const OpenMesh::Vec3f&)>::iterator it = pts.find(pt);

				if(it != pts.end())
				{
					vertexLookUp[i1] = it->second;
				}
				else
				{
					pts.insert(std::make_pair(pt, cnt));
					new_verts.push_back(pt);
					vertexLookUp[i1] = cnt;
					cnt++;
				}
			}

			// Update triangles
			for(std::vector<unsigned int>::iterator it = tris.begin(); it != tris.end(); it++)
			{
				*it = vertexLookUp[*it];
			}

			std::cerr << "Removed " << numV-cnt << " duplicate vertices of " << numV << " vertices" << std::endl;
			verts = std::vector<OpenMesh::Vec3f>(new_verts.begin(), new_verts.end());

			return cnt;
		}


		void removeDuplicateVertices(Mesh& mesh) {

			std::vector<unsigned int> indices;
			std::vector<OpenMesh::Vec3f> vertices;
			//iterate over all faces
			for (unsigned int i = 0; i < mesh.n_faces(); i++) {
				OpenMesh::FaceHandle fh(i);
				//iterate over all vertices in a face
				for (Mesh::FaceVertexIter fv_it = mesh.fv_begin(fh); fv_it!=mesh.fv_end(fh); fv_it++) {
					vertices.push_back(mesh.point(fv_it.handle()));
				}
				indices.push_back(3 * i + 0); indices.push_back(3 * i + 1); indices.push_back(3 * i + 2);
			}

			// Create an indexed face set out of the marching cube triangle soup.
			unsigned int cnt = removeDuplicateVertices(vertices, indices);

			// Convert indexed face set triangle mesh into directed edge data structure.
			Mesh n;
			Mesh::VertexHandle* vhandle =  new Mesh::VertexHandle[cnt];

			for(unsigned int i = 0; i<cnt; i++)	{
				vhandle[i] = n.add_vertex(vertices[i]);
			}

			std::vector<Mesh::VertexHandle> face_vhandles;
			std::cout << indices.size()/3 << std::endl;
			std::cout << vertices.size() << std::endl;
			for(unsigned int i = 0; i<indices.size()/3; i++)
			{
				if(indices[3*i+0] >= cnt ||indices[3*i+1] >= cnt || indices[3*i+2] >= cnt)
				{
					std::cout << "error" << std::endl;
					getchar();
				}

				// Check if triangle is degenerated	
				if((vhandle[indices[3*i+0]] != vhandle[indices[3*i+1]]) && (vhandle[indices[3*i+1]] != vhandle[indices[3*i+2]]) && (vhandle[indices[3*i+0]] != vhandle[indices[3*i+2]]))
				{
					face_vhandles.clear();
					face_vhandles.push_back(vhandle[indices[3*i+0]]);
					face_vhandles.push_back(vhandle[indices[3*i+1]]);
					face_vhandles.push_back(vhandle[indices[3*i+2]]);
					n.add_face(face_vhandles);
				}
			}

			delete [] vhandle;

			mesh = n;
		}
		*/
		

		static void convertToOpenMesh(const TriMeshf& triMesh, Mesh& out, bool keepVertexAttributes = false) {
			out.clear();

			if (keepVertexAttributes == false) {
				std::vector< Mesh::VertexHandle > vHandles;
				vHandles.reserve(triMesh.getVertices().size());

				//TODO ADD NORMALS

				for (const auto& v : triMesh.getVertices()) {
					vHandles.push_back(out.add_vertex(Mesh::Point(v.position.x, v.position.y, v.position.z)));
					//vHandles.push_back(out.add_vertex(
					//	Mesh::Point(v.position.x, v.position.y, v.position.z),
					//	Mesh::Normal(v.normal.x, v.normal.y, v.normal.z),
					//	Mesh::Color(v.color.x, v.color.y, v.color.z, v.color.w),
					//	Mesh::TexCoord2D(v.texCoord.x, v.texCoord.y)
					//	));	
				}

				for (const auto& f : triMesh.getIndices()) {
					out.add_face(vHandles[f.x], vHandles[f.y], vHandles[f.z]);
				}
			} else {
				throw MLIB_EXCEPTION("TODO implement");
			}
		}

		static void convertToTriMesh(Mesh& mesh, TriMeshf& out, bool keepVertexAttributes = false) {

			MeshDataf mData;
			if (keepVertexAttributes == false) {
				mData.m_Vertices.resize(mesh.n_vertices());
				for (unsigned int i = 0; i < mesh.n_vertices(); i++) {
					const auto& v = mesh.point(Mesh::VertexHandle(i));
					mData.m_Vertices[i] = vec3f(v.data()[0], v.data()[1], v.data()[2]);
				}

				mData.m_FaceIndicesVertices.resize(mesh.n_faces());
				for (unsigned int i = 0; i < mesh.n_faces(); i++) {
					const auto& f = mesh.face(Mesh::FaceHandle(i));
					auto iter = mesh.fv_iter(Mesh::FaceHandle(i));
					mData.m_FaceIndicesVertices[i][0] = iter->idx();
					iter++;									
					mData.m_FaceIndicesVertices[i][1] = iter->idx();
					iter++;									
					mData.m_FaceIndicesVertices[i][2] = iter->idx();			
				}
			} else {
				throw MLIB_EXCEPTION("TODO implement");
			}
			out = TriMeshf(mData);
		}

		//! decimates a mesh to a specific target vertex number; REVOMES ALSO ALL COLOR/NORMAL/TEXCOORD data
		static void decimate(TriMeshf& triMesh, size_t targetNumVertices, bool keepVertexAttributes = false) {

			Mesh        mesh;									// a mesh object
			convertToOpenMesh(triMesh, mesh, keepVertexAttributes);
			//std::cout << "size before: " << mesh.n_vertices() << std::endl;
			//removeDuplicateVertices(mesh);
			//std::cout << "size after: " << mesh.n_vertices() << std::endl;

			Decimater   decimater(mesh);						// a decimater object, connected to a mesh
			HModQuadric hModQuadric;							// use a quadric module
			decimater.add( hModQuadric );						// register module at the decimater
			decimater.initialize();								// let the decimater initialize the mesh and the modules
			decimater.decimate_to(targetNumVertices);			// do decimation
			mesh.garbage_collection();
			convertToTriMesh(mesh, triMesh, keepVertexAttributes);
			
		}

	}	//namespace OpenMeshTriMesh
}	//namespace ml

#endif	// EXT_OPENMESH_TRIMESH_H_