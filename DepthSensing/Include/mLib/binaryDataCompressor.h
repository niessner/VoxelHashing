
#ifndef CORE_UTIL_BINARYDATACOMPRESSOR_H_
#define CORE_UTIL_BINARYDATACOMPRESSOR_H_

namespace ml
{


//! interface to compress data
class BinaryDataCompressorInterface {
public:
	virtual void compressStreamToMemory(const BYTE *decompressedStream, UINT64 decompressedStreamLength, std::vector<BYTE> &compressedStream) const = 0;
	virtual void decompressStreamFromMemory(const BYTE *compressedStream, UINT64 compressedStreamLength, BYTE *decompressedStream, UINT64 decompressedStreamLength) const = 0;

	virtual std::string getTypename() const = 0;
};

//! interface to compress data
class BinaryDataCompressorNone : public BinaryDataCompressorInterface {
public:
	void compressStreamToMemory(const BYTE *decompressedStream, UINT64 decompressedStreamLength, std::vector<BYTE> &compressedStream) const {
		MLIB_ASSERT(false);	//just a dummy; should never come here
	}
	void decompressStreamFromMemory(const BYTE *compressedStream, UINT64 compressedStreamLength, BYTE *decompressedStream, UINT64 decompressedStreamLength) const {
		MLIB_ASSERT(false); //just a dummy; should never come here
	}

	std::string getTypename() const {
		return "no compression";
	}

};

//! interface to compress data
class BinaryDataCompressorDefault : public BinaryDataCompressorInterface {
public:
	void compressStreamToMemory(const BYTE *decompressedStream, UINT64 decompressedStreamLength, std::vector<BYTE> &compressedStream) const {
		compressedStream.resize(decompressedStreamLength);
		memcpy(&compressedStream[0], decompressedStream, decompressedStreamLength);
	}

	void decompressStreamFromMemory(const BYTE *compressedStream, UINT64 compressedStreamLength, BYTE *decompressedStream, UINT64 decompressedStreamLength) const {
		memcpy(decompressedStream, compressedStream, compressedStreamLength);
	}

	std::string getTypename() const {
		return "stupid copying - should not be used";
	}
};

}  // namespace ml

#endif  // CORE_UTIL_BINARYDATACOMPRESSOR_H_
