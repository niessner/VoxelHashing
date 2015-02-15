#ifndef EXT_BOOST_SERIALIZATION_H_
#define EXT_BOOST_SERIALIZATION_H_

// Get rid of warning due to ambiguity in pre-XP vs XP and later Windows versions
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0501
#endif

#include <boost/serialization/array.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <boost/asio/streambuf.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/traits.hpp>


namespace ml {

class InOutArchive {
  public:
	InOutArchive()
		: m_buf()
		, m_os(&m_buf, std::ios::binary)
		, m_is(&m_buf, std::ios::binary)
		, m_oa(m_os)
		, m_ia(m_is) { }

	template<typename T>
	inline InOutArchive& operator<<(T const& obj) {
		m_oa << obj;
		return *this;
	}

	template<typename T>
	inline InOutArchive& operator>>(T& obj) {
		m_ia >> obj;
		return *this;
	}

  private:
	boost::asio::streambuf m_buf;
	std::ostream m_os;
	std::istream m_is;
	boost::archive::binary_oarchive m_oa;
	boost::archive::binary_iarchive m_ia;
};

}  // namespace ml

namespace boost {
namespace serialization {

    /*template<class Archive, class T>
    void save(Archive & ar, const ml::point2d<T>& p, const unsigned int version)
    {
        ar & p.array;
    }
    template<class Archive, class T>
    void load(Archive & ar, ml::point2d<T>& p, const unsigned int version)
    {
        ar & p.array;
        //ar & p.x & p.y;
    }

    template<class Archive, class T>
    void save(Archive & ar, const ml::point3d<T>& p, const unsigned int version)
    {
        ar & p.array;
    }
    template<class Archive, class T>
    void load(Archive & ar, ml::point3d<T>& p, const unsigned int version)
    {
        ar & p.array;
        //ar & p.x & p.y & p.z;
    }

    template<class Archive, class T>
    void save(Archive & ar, const ml::point4d<T>& p, const unsigned int version)
    {
        ar & p.array;
    }
    template<class Archive, class T>
    void load(Archive & ar, ml::point4d<T>& p, const unsigned int version)
    {
        ar & p.array;
        //ar & p.x & p.y & p.z & p.w;
    }

    template<class Archive, class T>
    inline void serialize(Archive& ar, ml::point2d<T>& p, const unsigned int version) {
        boost::serialization::split_free(ar, p, version);
    }

    template<class Archive, class T>
    inline void serialize(Archive& ar, ml::point3d<T>& p, const unsigned int version) {
        boost::serialization::split_free(ar, p, version);
    }

    template<class Archive, class T>
    inline void serialize(Archive& ar, ml::point4d<T>& p, const unsigned int version) {
        boost::serialization::split_free(ar, p, version);
    }*/
/*#else // MLIB_USE_ARRAY_SERIALIZATION

template<class Archive, class T>
inline void serialize(Archive& ar, ml::point2d<T>& p, const unsigned int version) {
    ar & p.x & p.y;
}

template<class Archive, class T>
inline void serialize(Archive& ar, ml::point3d<T>& p, const unsigned int version) {
    ar & p.x & p.y & p.z;
}

template<class Archive, class T>
inline void serialize(Archive& ar, ml::point4d<T>& p, const unsigned int version) {
    ar & p.x & p.y & p.z & p.w;
}

#endif // MLIB_USE_ARRAY_SERIALIZATION*/

//
// In the next version of mlib, this should definitely be switched to the ar & p.x & p.y style
// a 100x100 sphere in an archive is 7MB in the array style, but 4.2MB in the primitive style
//

template<class Archive, class T>
inline void serialize(Archive& ar, ml::point2d<T>& p, const unsigned int version) {
    ar & p.array;
}

template<class Archive, class T>
inline void serialize(Archive& ar, ml::point3d<T>& p, const unsigned int version) {
	ar & p.array;
}

template<class Archive, class T>
inline void serialize(Archive& ar, ml::point4d<T>& p, const unsigned int version) {
  ar & p.array;
}



template<class Archive, class T>
inline void serialize(Archive& ar, ml::Matrix4x4<T>& m, const unsigned int version) {
  ar & m.matrix;
}

template<class Archive, class T>
inline void serialize(Archive& ar, ml::OrientedBoundingBox3<T>& b, const unsigned int version) {
    vec3f v[4];
    v[0] = b.getAnchor();
    v[1] = b.getAxisX();
    v[2] = b.getAxisY();
    v[3] = b.getAxisZ();
    ar & v[0] & v[1] & v[2] & v[3];

    //
    // TODO: this should be broken into save-load style or friended somehow. It's silly to modify the bounding box when saving.
    //
    b = ml::OBBf(v[0], v[1], v[2], v[3]);
}

template<class Archive>
inline void serialize(Archive& ar, ml::TriMesh<float>::Vertex<float>& v, const unsigned int version) {
    ar & v.position & v.normal & v.color & v.texCoord;
}

template<class Archive, class T>
inline void serialize(Archive& ar, ml::Material<T>& m, const unsigned int version) {
    ar & m.m_name & m.m_ambient & m.m_diffuse & m.m_specular & m.m_shiny & m.m_TextureFilename_Ka & m.m_TextureFilename_Kd & m.m_TextureFilename_Ks;
}

}  // namespace serialization
}  // namespace boost

// TODO: Move this to an appropriate test file
static void testBoostSerialization() {
	namespace io = boost::iostreams;
	namespace arc = boost::archive;

	// Make objects
	const int N = 10000;
	std::vector<ml::vec3f> ps;
	for (int i = 0; i < N; i++) { ps.push_back(ml::vec3f(static_cast<float>(i), 2, 3)); }
	const ml::vec3uc v(0, 255, 0);

	// TEXT ARCHIVE TEST
	std::string txt_file("demofile.txt");
	// save to file
	{
		std::ofstream ofs(txt_file);
		arc::text_oarchive oa(ofs);
		oa << ps;
		oa << v;
	}
	// load from file
	std::vector<ml::vec3f> ps2;
	ml::vec3uc v2;
	{
		std::ifstream ifs(txt_file);
		arc::text_iarchive ia(ifs);
		ia >> ps2;
		ia >> v2;
	}

	// IN-MEMORY BUFFER TEST
	boost::asio::streambuf buf;
	std::vector<ml::vec3f> ps3;
	ml::vec3uc v3;
	// save to buf
	{
		std::ostream os(&buf, std::ios::binary);
		arc::binary_oarchive oa(os);
		oa << ps;
		oa << v;
	}
	// load from buf
	{
		std::istream is(&buf, std::ios::binary);
		arc::binary_iarchive ia(is);
		ia >> ps3;
		ia >> v3;
	}

	// IN-MEMORY SAVE-LOAD BUFFER TEST
	boost::asio::streambuf buf2;
	std::vector<ml::vec3f> ps7;
	ml::vec3uc v7;
	{
		std::ostream os(&buf2, std::ios::binary);
		arc::binary_oarchive oa(os);
		std::istream is(&buf2, std::ios::binary);
		arc::binary_iarchive ia(is);
		oa << ps;
		ia >> ps7;
		oa << v;
		ia >> v7;
	}

	// IN-MEMORY COMPRESSED BUFFER
	boost::asio::streambuf buf3;
	io::filtering_ostreambuf out;
	out.push(io::zlib_compressor());
	out.push(buf3);
	{
		arc::binary_oarchive oa(out);
		oa << ps;
	}
	io::close(out);

	io::filtering_istreambuf in;
	in.push(io::zlib_decompressor());
	in.push(buf3);
	{
		ps.clear();
		arc::binary_iarchive ia(in);
		ia >> ps;
	}
	io::close(in);

	// BINARY FILE TEST
	std::vector<ml::vec3f> ps4;
	ml::vec3uc v4;
	std::string bin_file("demofile.bin");
	// Write out to binary file
	std::ofstream ofs(bin_file, std::ios::binary);
	{
		arc::binary_oarchive oa(ofs);
		oa << ps;
		oa << v;
	}
	ofs.close();

	// load back from binary file
	std::ifstream ifs(bin_file, std::ios::binary);
	{
		arc::binary_iarchive ia(ifs);
		ia >> ps4;
		ia >> v4;
	}
	ifs.close();

	// COMPRESSED BINARY FILE TEST
	std::vector<ml::vec3f> ps5;
	ml::vec3uc v5;
	std::string binz_file("demofile.bin.z");
	// Write out to binary file
	std::ofstream ofsz(binz_file, std::ios::out | std::ios::binary);
	{
		io::filtering_streambuf<io::output> out;
		out.push(io::zlib_compressor(io::zlib::best_compression));
		out.push(ofsz);
		arc::binary_oarchive oa(out);
		oa << ps;
		oa << v;
	}
	ofsz.close();

	// load back from binary file
	std::ifstream ifsz(binz_file, std::ios::in | std::ios::binary);
	{
		io::filtering_streambuf<io::input> in;
		in.push(io::zlib_decompressor());
		in.push(ifsz);
		arc::binary_iarchive ia(in);
		ia >> ps5;
		ia >> v5;
	}
	ifsz.close();

	// Custom in-out archive test
	ml::InOutArchive inout;
	inout << ps;
	inout << v;
	ps.clear();
	v2.y = 0;
	inout >> ps;
	inout >> v2;
}

#endif  // EXT_BOOST_SERIALIZATION_H_
