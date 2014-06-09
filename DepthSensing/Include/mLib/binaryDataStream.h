
#ifndef CORE_UTIL_BINARYDATASTREAM_H_
#define CORE_UTIL_BINARYDATASTREAM_H_

namespace ml
{

template<class BinaryDataBuffer, class BinaryDataCompressor>
class BinaryDataStream {
public:
	BinaryDataStream() {}
	BinaryDataStream(const std::string& filename, bool clearStream) {
		m_DataBuffer.openBufferStream(filename, clearStream);
	}
	//! only required for files
	void openStream(const std::string& filename, bool clearStream) {
		m_DataBuffer.openBufferStream(filename, clearStream);
	}

	void closeStream() {
		m_DataBuffer.closeBufferStream();
	}

	template <class T>
	void writeData(const T& t) {
		writeData((const BYTE*)&t, sizeof(T));
	}

	//start compression after that byte size (only if compression is enabled)
#define COMPRESSION_THRESHOLD_ 1024 
	void writeData(const BYTE* t, size_t size) {
		const bool useCompression = !std::is_same<BinaryDataCompressorNone, BinaryDataCompressor>::value;
		if (useCompression && size > COMPRESSION_THRESHOLD_) {
			std::vector<BYTE> compressedT;
			//ZLibWrapper::CompressStreamToMemory(t, size, compressedT, false);
			m_DataCompressor.compressStreamToMemory(t, size, compressedT);
			UINT64 compressedSize = compressedT.size();
			m_DataBuffer.writeData((const BYTE*)&compressedSize, sizeof(compressedSize));
			m_DataBuffer.writeData(&compressedT[0], compressedSize);
		} else 	{
			m_DataBuffer.writeData(t, size);
		}

	}

	template <class T>
	void readData(T* result) {
		readData((BYTE*)result, sizeof(T));
	}

	void readData(BYTE* result, size_t size) {
		const bool useCompression = !std::is_same<BinaryDataCompressorNone, BinaryDataCompressor>::value;
		if (useCompression && size > COMPRESSION_THRESHOLD_) {
			UINT64 compressedSize;
			m_DataBuffer.readData((BYTE*)&compressedSize, sizeof(UINT64));
			std::vector<BYTE> compressedT;	compressedT.resize(compressedSize);
			m_DataBuffer.readData(&compressedT[0], compressedSize);
			//ZLibWrapper::DecompressStreamFromMemory(&compressedT[0], compressedSize, result, size);
			m_DataCompressor.decompressStreamFromMemory(&compressedT[0], compressedSize, result, size);
		} else 
		{
			m_DataBuffer.readData(result, size);
		}
	}

	//! clears the read offset: copies all data to the front of the data array and frees all unnecessary memory
	void clearReadOffset() {
		m_DataBuffer.clearReadOffset();
	}

	//! destroys all the data in the stream (DELETES ALL DATA!)
	void clearStream() {
		m_DataBuffer.clearBuffer();
	}

	//! saves the stream to file; does not affect current data
	void saveToFile(const std::string &filename) {
		m_DataBuffer.saveToFile(filename);
	}

	//! loads a binary stream from file; destorys all previous data in the stream
	void loadFromFile(const std::string &filename) {
		m_DataBuffer.loadFromFile(filename);
	} 

	void reserve(size_t size) {
		m_DataBuffer.reserve(size);
	}

	void flush() {
		m_DataBuffer.flushBufferStream();
	}
private:
	BinaryDataBuffer		m_DataBuffer;
	BinaryDataCompressor	m_DataCompressor;
};

typedef BinaryDataStream<BinaryDataBufferMemory, BinaryDataCompressorNone> BinaryDataStreamVector;
typedef BinaryDataStream<BinaryDataBufferFile, BinaryDataCompressorNone> BinaryDataStreamFile;
//typedef BinaryDataStream<BinaryDataBufferVector, BinaryDataCompressorDefault> BinaryDataStreamCompressedVector; (see zlib for instance)
//typedef BinaryDataStream<BinaryDataBufferFile, BinaryDataCompressorDefault> BinaryDataStreamCompressedFile; (see zlib for instance)

//////////////////////////////////////////////////////
/////////write stream operators for base types ///////
//////////////////////////////////////////////////////

//cannot overload via template since it is not supposed to work for complex types

template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, UINT64 i) {
	s.writeData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, int i) {
	s.writeData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, unsigned int i) {
	s.writeData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, short i) {
	s.writeData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, unsigned short i) {
	s.writeData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, char i) {
	s.writeData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, unsigned char i) {
	s.writeData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, float i) {
	s.writeData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, double i) {
	s.writeData(i);
	return s;
}

template<class BinaryDataBuffer, class BinaryDataCompressor, class T>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const std::vector<T>& v) {
	s << (UINT64)v.size();
	s.reserve(sizeof(T)*v.size());
	for (size_t i = 0; i < v.size(); i++) {
		s << v[i];
	}
	return s;
}

template<class BinaryDataBuffer, class BinaryDataCompressor, class T>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const Grid2D<T>& g) {
	s << (UINT64)g.rows() << (UINT64)g.cols();
	s.reserve(sizeof(T) * g.rows() * g.cols());
	for (UINT64 row = 0; row < g.rows(); row++)
		for (UINT64 col = 0; col < g.cols(); col++)
			s << g(row, col);
	return s;
}

template<class BinaryDataBuffer, class BinaryDataCompressor, class T>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const std::list<T>& l) {
	s << (UINT64)l.size();
	s.reserve(sizeof(T)*l.size());
	for (typename std::list<T>::const_iterator iter = l.begin(); iter != l.end(); iter++) {
		s << *l;
	}
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const std::string& l) {
	s << (UINT64)l.size();
	s.reserve(sizeof(char)*l.size());
	for (size_t i = 0; i < l.size(); i++) {
		s << l[i];
	}
	return s;
}





//////////////////////////////////////////////////////
/////////read stream operators for base types ///////
//////////////////////////////////////////////////////

//cannot overload via template since it is not supposed to work for complex types
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, UINT64& i) {
	s.readData(&i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, int& i) {
	s.readData(&i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, unsigned int& i) {
	s.readData(&i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, short& i) {
	s.readData(&i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, unsigned short& i) {
	s.readData(&i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, char& i) {
	s.readData(&i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, unsigned char& i) {
	s.readData(&i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, float& i) {
	s.readData(&i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, double& i) {
	s.readData(&i);
	return s;
}

template<class BinaryDataBuffer, class BinaryDataCompressor, class T>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, std::vector<T>& v) {
	UINT64 size;
	s >> size;
	v.resize(size);
	for (size_t i = 0; i < v.size(); i++) {
		s >> v[i];
	}
	return s;
}

template<class BinaryDataBuffer, class BinaryDataCompressor, class T>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, Grid2D<T>& g) {
	UINT64 rows, cols;
	s >> rows >> cols;
	g.allocate(rows, cols);
	for (UINT64 row = 0; row < g.rows(); row++)
		for (UINT64 col = 0; col < g.cols(); col++)
			s << g(row, col);
	return s;
}

template<class BinaryDataBuffer, class BinaryDataCompressor, class T>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const std::list<T>& l) {
	UINT64 size;
	s >> size;
	l.clear();
	for (size_t i = 0; i < size; i++) {
		T curr;
		s >> curr;
		l.push_back(l);
	}
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, std::string& v) {
	UINT64 size;
	s >> size;
	v.resize(size);
	for (size_t i = 0; i < v.size(); i++) {
		s >> v[i];
	}
	return s;
}

}  // namespace ml

#endif  // CORE_UTIL_BINARYDATASTREAM_H_
