
#ifndef CORE_UTIL_BINARYDATASERIALIZE_H_
#define CORE_UTIL_BINARYDATASERIALIZE_H_

#include "BinaryDataStream.h"

namespace ml
{

template<class ChildClass>
class BinaryDataSerialize {
public:
	unsigned int getSizeInBytes() {
		return sizeof(ChildClass);
	}
};

template<class BinaryDataBuffer, class BinaryDataCompressor, class ChildClass>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<< (BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const BinaryDataSerialize<ChildClass>& o) {		
	s.writeData(*(const ChildClass*)&o);	//cast it to the child class to get the right size
	return s;
} 

template<class BinaryDataBuffer, class BinaryDataCompressor, class ChildClass>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>> (BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, BinaryDataSerialize<ChildClass> &o) {
	s.readData((ChildClass*)&o);
	return s;
}

}  // namespace ml

#endif  // CORE_UTIL_BINARYDATASERIALIZE_H__
