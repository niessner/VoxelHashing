
#ifndef CORE_UTIL_BINARYDATABUFFER_H_
#define CORE_UTIL_BINARYDATABUFFER_H_

#include <vector>
#include <list>
#include <fstream>
#include <string>

namespace ml
{

/////////////////////////////////////////////////////////////
// BinaryDataBuffers (class used by BINARY_DATA_STREAM)    //
/////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////
// BinaryDataBuffers (one for file, one for system memory) //
/////////////////////////////////////////////////////////////

class BinaryDataBufferFile {
public:
	BinaryDataBufferFile() {
		m_readOffset = 0;
		m_fileSize = 0;
	}

	void openBufferStream(const std::string& filename, bool clearBuffer = false) {
		m_filename = filename;
		if (clearBuffer)	remove(m_filename.c_str());
		closeFileStream();
		openFileStream();
	}
	void closeBufferStream() {
		closeFileStream();
	}

	~BinaryDataBufferFile() {
		closeFileStream();
		if (m_readOffset == 0 && m_fileSize == 0) {
			remove(m_filename.c_str());
		}
	}

	void writeData(const BYTE* t, size_t size) {
		//Console::log() << "tellp() " << m_fileStream.tellp() << std::endl;
		m_fileStream.seekp(m_fileSize);	//always append at the end
		m_fileStream.write((char*)t, size);
		m_fileSize += size;
	}

	void readData(BYTE* result, size_t size) {
		//Console::log() << "tellg() " << m_fileStream.tellg() << std::endl;
		//assert(m_readOffset + size <= m_fileSize);
		if (m_readOffset + size > m_fileSize) throw MLIB_EXCEPTION("invalid read; probably wrong file name (" + m_filename + ")?");
		m_fileStream.seekg(m_readOffset);
		m_fileStream.read((char*)result, size);
		m_readOffset += size;
	}


	//! destroys all the data in the stream
	void clearBuffer() {
		closeFileStream();
		remove(m_filename.c_str());
		openFileStream();
		MLIB_ASSERT_STR(m_fileSize == 0, "file buffer not cleared correctly");
		m_readOffset = 0;
	}

	void clearReadOffset() {
		size_t len = m_fileSize - m_readOffset;
		if (len == 0) {  //if there is no data left, clear the buffer
			clearBuffer();
		} else {
			std::vector<BYTE> oldData;
			copyDataToMemory(oldData);

			closeFileStream();
			remove(m_filename.c_str());
			openFileStream();
			m_fileStream.write((char*)&oldData[0], oldData.size());
			m_readOffset = 0;
			MLIB_ASSERT_STR(m_fileSize == oldData.size(), "");
		}
	}

	void reserve(size_t size) {
		return;	//doesn't make much sense for files, does it?
	}

	void saveToFile(const std::string &filename) {
		std::vector<BYTE> oldData;
		copyDataToMemory(oldData);

		std::ofstream output(filename, std::ios::binary);
		output.write((char*)&oldData[0], sizeof(BYTE)*oldData.size());
		if (!output.is_open())	throw MLIB_EXCEPTION(filename);
		output.close();
		return;
	}

	//! loads a binary stream from file; destorys all previous data in the stream
	void loadFromFile(const std::string &filename) {
		m_fileStream.close();

        // TODO: replace this with a utility function
		size_t inputFileSize = util::getFileSize(filename);

		BYTE* data = new BYTE[inputFileSize];
		std::ifstream input(filename, std::ios::binary);
		if (!input.is_open())	throw MLIB_EXCEPTION(filename);
		input.read((char*)data, sizeof(BYTE)*inputFileSize);
		input.close();

		clearBuffer();	//clear the old values
		m_fileStream.write((char*)data, sizeof(BYTE)*inputFileSize);
		MLIB_ASSERT(m_fileSize == inputFileSize);
		m_readOffset = 0;
	}

	//! flushes the stream
	void flushBufferStream() {
		m_fileStream.flush();
	}
private:


	//! reads all the 'active' file data to system memory
	void copyDataToMemory(std::vector<BYTE>& data) {
		size_t len = m_fileSize - m_readOffset;
		data.resize(len);
		m_fileStream.seekg(m_readOffset);
		m_fileStream.read((char*)&data[0], sizeof(BYTE)*len);
	}

	//! opens the file stream
	void openFileStream() {
		if (m_fileStream.is_open())	m_fileStream.close();

		if (!util::fileExists(m_filename))	m_fileSize = 0;	//in case there was no file before
		else m_fileSize = util::getFileSize(m_filename);

		m_fileStream.open(m_filename.c_str(), std::ios::binary | std::ios::out | std::ios::in);
		if (!m_fileStream.is_open()) {
			m_fileStream.open(m_filename.c_str(), std::ios::binary | std::ios::out);
			m_fileStream.close();
			m_fileStream.open(m_filename.c_str(), std::ios::binary | std::ios::out | std::ios::in);
		} 
		if (!m_fileStream.is_open() || !m_fileStream.good()) throw MLIB_EXCEPTION(m_filename);
	}

	//! closes the file stream; data is automatically saved...
	void closeFileStream() {
		if (m_fileStream.is_open())	m_fileStream.close();
	}

	std::string		m_filename;
	std::fstream	m_fileStream;
	size_t			m_readOffset;
	size_t			m_fileSize;
};






class BinaryDataBufferMemory {
public:
	BinaryDataBufferMemory() {
		m_readOffset = 0;
	}
	void openBufferStream(const std::string& filename, bool clearBuffer = false) {
		MLIB_ASSERT(false);
		//dummy just needed for file stream
		return;
	}
	void closeBufferStream() {
		MLIB_ASSERT(false);
		//dummy just needed for file stream
		return;
	}

	void writeData(const BYTE* t, size_t size) {
		size_t basePtr = m_Data.size();
		m_Data.resize(basePtr + size);
		memcpy(&m_Data[0] + basePtr, t, size);
	}

	void readData(BYTE* result, size_t size) {
		MLIB_ASSERT(m_readOffset + size <= m_Data.size());

		memcpy(result, &m_Data[0] + m_readOffset, size);
		m_readOffset += size;

		//free memory if we reached the end of the stream
		if (m_readOffset == m_Data.size()) {
			m_Data.resize(0);
			m_readOffset = 0;
		}
	}


	//! destroys all the data in the stream
	void clearBuffer() {
		m_Data.clear();
		m_readOffset = 0;
	}

	void clearReadOffset() {
		size_t len = m_Data.size() - m_readOffset;
		for (unsigned int i = 0; i < len; i++) {
			m_Data[i] = m_Data[i + m_readOffset];
		}
		m_Data.resize(len);
		m_readOffset = 0;
	}

	void reserve(size_t size) {
		if (size > m_Data.size())
			m_Data.reserve(size);
	}

	void saveToFile(const std::string &filename) {
		std::ofstream output(filename, std::ios::binary);
		output.write((char*)&m_Data[0], sizeof(BYTE)*m_Data.size());
		if (!output.is_open())	throw MLIB_EXCEPTION(filename);
		output.close();
	}


	//! returns the file size; if the file cannot be opened returns -1 (e.g., the file does not exist)
	size_t getFileSizeInBytes(const std::string &filename) {
		std::ifstream file(filename, std::ios::binary | std::ios::ate);
		if (!file.is_open())	return -1;
		size_t size = file.tellg();
		file.close();
		return size;
	}

	//! loads a binary stream from file; destorys all previous data in the stream
	void loadFromFile(const std::string &filename) {
		size_t inputFileSize = getFileSizeInBytes(filename);
		m_Data.resize(inputFileSize);
		std::ifstream input(filename, std::ios::binary);
		if (!input.is_open())	throw MLIB_EXCEPTION(filename);
		input.read((char*)&m_Data[0], sizeof(BYTE)*inputFileSize);
		input.close();
		m_readOffset = 0;
	} 

	//! since all writes are immediate, there is nothing to do
	void flushBufferStream() {
		return;
	}
private:
	std::vector<BYTE>	m_Data;
	size_t				m_readOffset;
};

}  // namespace ml

#endif  // CORE_UTIL_BINARYDATABUFFER_H_
