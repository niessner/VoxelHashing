
#ifndef _BASEIMAGE_H_
#define _BASEIMAGE_H_

#include "baseImageHelper.h"

namespace ml {

template <class T>
class BaseImage {
public:
	BaseImage() {
		m_Data = NULL;
		m_Height = m_Width = 0;
	}

	BaseImage(unsigned int height, unsigned int width, const T *data = NULL) {
		create(height, width);

		if (data) {
			memcpy(m_Data, data, sizeof(T) * height * width);
		}
	}

	//! Copy constructor
	BaseImage(const BaseImage& other) {
		create(other.m_Height, other.m_Width);
		memcpy(m_Data, other.m_Data, sizeof(T) * m_Height * m_Width);
		m_InvalidValue = other.getInvalidValue();
	}

	//! Move constructor
	BaseImage(BaseImage&& other) {
		m_Data = NULL;
		m_Height = m_Width = 0;
		swap(*this, other);
	}

	//! Copy constructor for other classes
	template<class U>
	BaseImage(const BaseImage<U>& other) {
		create(other.getHeight(), other.getWidth());
		for (unsigned int i = 0; i < m_Width*m_Height; i++) {
			BaseImageHelper::convertBaseImagePixel<T,U>(m_Data[i], other.getDataPointer()[i]);
		}
		const U& otherInvalidValue = other.getInvalidValue();
		BaseImageHelper::convertBaseImagePixel<T,U>(m_InvalidValue, otherInvalidValue);
	}

	//! adl swap
	friend void swap(BaseImage& a, BaseImage& b) {
		std::swap(a.m_Data, b.m_Data);
		std::swap(a.m_Width, b.m_Width);
		std::swap(a.m_Height, b.m_Height);
		std::swap(a.m_InvalidValue, b.m_InvalidValue);
	}

	void initialize(const T *data = NULL)
	{
		if (data) {
			memcpy(m_Data, data, sizeof(T) * m_Width * m_Height);
		}
	}

	~BaseImage(void) {
		SAFE_DELETE_ARRAY(m_Data);
	}

	//! Returns the difference of two images (current - other)
	BaseImage<T> operator-(const BaseImage<T> &other) const {
		if (other.m_Width != m_Width || other.m_Height != m_Height)	throw EXCEPTION("Invalid image dimensions");
		BaseImage<T> im(m_Height, m_Width);
		for (unsigned int i = 0; i < m_Height * m_Width; i++) {
			im.m_Data[i] = m_Data[i] - other.m_Data[i];
		}
		return im;
	}
	//! Returns the sum of two images (current + other)
	BaseImage<T> operator+(const BaseImage<T> &other) const {
		if (other.m_Width != m_Width || other.m_Height != m_Height)	throw EXCEPTION("Invalid image dimensions");
		BaseImage<T> im(m_Height, m_Width);
		for (unsigned int i = 0; i < m_Height * m_Width; i++) {
			im.m_Data[i] = m_Data[i] + other.m_Data[i];
		}
		return im;
	}

	//! Mutator Operator (unsigned int)
	T& operator()(unsigned int y, unsigned int x) {
		assert(y < m_Height && x < m_Width);
		return m_Data[y*m_Width + x];
	}

	//! Mutator Operator (int)
	T& operator()(int y, int x) {
		assert((unsigned int)y < m_Height && (unsigned int)x < m_Width);
		return m_Data[y*m_Width + x];
	}

	//! Mutator Operator (double)
	T& operator()(double y, double x) {
		return (*this)((unsigned int)math::round(y*(m_Height-1)), (unsigned int)math::round(x*(m_Width-1)));
	}

	//! Mutator Operator (float)
	T& operator()(float y, float x) {
		return (*this)((unsigned int)math::round(y*(m_Height-1)), (unsigned int)math::round(x*(m_Width-1)));
	}

	template <class S>
	void setPixel(S y, S x, const T& value) {
		(*this)(y,x) = value;
	}

	//! Access Operator (unsigned int)
	const T& operator()(unsigned int y, unsigned int x) const {
		assert(y < m_Height && x < m_Width);
		return m_Data[y*m_Width + x];
	}

	//! Access Operator (int)
	const T& operator()(int y, int x) const {
		assert((unsigned int)y < m_Height && (unsigned int)x < m_Width);
		return m_Data[y*m_Width + x];
	}

	//! Access Operator (double)
	const T& operator()(double y, double x) const {
		return (*this)((unsigned int)round(y*(m_Height-1)), (unsigned int)round(x*(m_Width-1)));
	}

	//! Access Operator (float)
	const T& operator()(float y, float x) const {
		return (*this)((unsigned int)round(y*(m_Height-1)), (unsigned int)round(x*(m_Width-1)));
	}

	//! Returns the Pixel value at that position (calls the function corresponding to the parameter type)
	template <class S>
	const T& getPixel(S y, S x) const {
		return (*this)(y,x);
	}


	//! Assignment operator
	BaseImage& operator=(const BaseImage& other) {
		if (this != &other) {
			if (m_Width != other.m_Width || m_Height != other.m_Height) {
				SAFE_DELETE_ARRAY(m_Data);
				create(other.m_Height, other.m_Width);
			}

			memcpy(m_Data, other.m_Data, sizeof(T) * m_Width * m_Height);
			m_InvalidValue = other.getInvalidValue();
		}
		return *this;
	}

	//! Assignment operator r-value
	BaseImage& operator=(BaseImage&& other) {
		swap(*this, other);
		return *this;
	}

	//! Comparison operator
	bool operator==(const BaseImage& other) {
		if (m_Width != other.m_Width || m_Height != other.m_Height)	return false;
		for (unsigned int i = 0; i < m_Height * m_Width; i++) {
			if (m_Data[i] != other.m_Data[i])	return false;
		}
		return true;
	}

	//! Allocates data so that the current image and other have the same size
	void allocateSameSize(const BaseImage& other) {
		if (m_Width != other.m_Width || m_Height != other.m_Height) {
			SAFE_DELETE_ARRAY(m_Data);
			create(other.m_Height, other.m_Width);
		}
	}

	//! Allocates the images to the given size
	void allocateToSize(unsigned int height, unsigned int width) {
		if (m_Width != width || m_Height != height) {
			SAFE_DELETE_ARRAY(m_Data);
			create(height, width);
		}
	}

	//! Copies a source image into a region of the current image
	void copyIntoImage(const BaseImage &source, unsigned int startY, unsigned int startX) {
		assert(source.getWidth() + startX <= getWidth() && source.getHeight() + startY <= getHeight());
		for (unsigned int y = startY; y < startY + source.getHeight(); y++) {
			for (unsigned int x = startX; x < startX + source.getWidth(); x++) {
				(*this)(y,x) = source(y-startY, x-startX);
			}
		}
	}

	//! Returns the width of the image
	unsigned int getWidth() const {
		return m_Width;
	}

	//! Returns the height of the image
	unsigned int getHeight() const {
		return m_Height;
	}

	//! Returns the image data (linearized array)
	const T* getDataPointer() const {
		return m_Data;
	}

	//! Returns the image data (linearized array)
	T* getDataPointer() {
		return m_Data;
	}


	//! saves a file to a binary depth image (.mbindepth) or a binary color image (.binRGB); could be any bytes per pixel
	void saveAsBinaryMImage(const std::string &filename) const {
		saveBinaryMImage(filename, m_Data, m_Height, m_Width);
	}

	//! saves a binary m image
	static void saveBinaryMImage(const std::string& filename, const void* data, unsigned int height, unsigned int width) {
		saveBinaryMImageArray(filename, &data, height, width, 1);
	}

	static void saveBinaryMImageArray(const std::string& filename, const std::vector<BaseImage<T>>& images) {
		assert(images.size() >= 1);
		void** data = new data*[images.size()];
		for (unsigned int i = 0; i < images.size(); i++) {
			assert(images[0].getWidth() == images[i].getWidth());
			assert(images[0].getHeight() == images[i].getHeight());
			data[i] = images[i].getDataPointer();
		}
		saveBinaryMImageArray(filename, data, images[0].getHeight(), images[0].getWidth(), images.size());
		SAFE_DELETE_ARRAY(data);
 	}

	//! saves an array of binary m images
	static void saveBinaryMImageArray(const std::string& filename, const void** data, unsigned int height, unsigned int width, unsigned int numImages = 1) {
		if (util::getFileExtension(filename) != "mbindepth" && util::getFileExtension(filename) != "mbinRGB") throw MLIB_EXCEPTION("invalid file extension" + util::getFileExtension(filename));

		std::ofstream file(filename, std::ios::binary);
		if (!file.is_open())	throw std::ios::failure(__FUNCTION__ + std::string(": could not open file ") + filename);

		unsigned int bytesPerPixel = sizeof(T);
		file.write((char*)&numImages, sizeof(unsigned int));
		file.write((char*)&height, sizeof(unsigned int));
		file.write((char*)&width, sizeof(unsigned int));
		file.write((char*)&bytesPerPixel, sizeof(unsigned int));
		for (unsigned int i = 0; i < numImages; i++) {
			file.write((char*)data[i], width * height * bytesPerPixel);
		}
		file.close();
	}

	//! loads a file from a binary depth image (.mbindepth) or a binary color image (.binRGB)
	void loadFromBinaryMImage(const std::string& filename) {
		loadBinaryMImage(filename, (void**)&m_Data, m_Height, m_Width);
	}

	static void loadBinaryMImage(const std::string& filename, void** data, unsigned int& height, unsigned int& width) {
		std::vector<void*> dataArray;
		loadBinaryMImageArray(filename, dataArray, height, width);
		assert(dataArray.size() == 1);
		if (*data)	delete[] data;
		*data = dataArray[0];
	}

	//! loads a binary array of m images
	static void loadBinaryMImageArray(const std::string& filename, std::vector<void*>& data, unsigned int& height, unsigned int& width) {
		if (util::getFileExtension(filename) != "mbindepth" && util::getFileExtension(filename) != "mbinRGB") throw MLIB_EXCEPTION("invalid file extension" + util::getFileExtension(filename));

		std::ifstream file(filename, std::ios::binary);
		if (!file.is_open())	throw std::ios::failure(__FUNCTION__ + std::string(": could not open file ") + filename);

		unsigned int bytesPerPixel, numImages;
		file.read((char*)&numImages, sizeof(unsigned int));
		file.read((char*)&height, sizeof(unsigned int));
		file.read((char*)&width, sizeof(unsigned int));
		file.read((char*)&bytesPerPixel, sizeof(unsigned int));
		assert(sizeof(T) == bytesPerPixel);
		for (unsigned int i = 0; i < numImages; i++) {
			void* currData = new T[width*height];
			file.read((char*)currData, width * height * bytesPerPixel);
			data.push_back(currData);
		}
		file.close();
	}

	//! counts the number of pixels not equal to value
	unsigned int getNumPixelsNotEqualTo(const T &value) {
		unsigned int count = 0;
		for (unsigned int i = 0; i < m_Height * m_Width; i++) {
			if (value != m_Data[i])	count++;
		}
		return count;
	}

	//! sets all pixels with a specific value (oldValue); to a new value (newValue)
	void setPixelsWithValueToNewValue(const T& oldValue, const T& newValue) {
		for (unsigned int i = 0; i < m_Height * m_Width; i++) {
			if (m_Data[i] == oldValue)	m_Data[i] = newValue;
		}
	}

	//! sets all pixels to value
	void clearImage(const T &value) {
		for (unsigned int i = 0; i < m_Height * m_Width; i++) {
			m_Data[i] = value;
		}
	}

	//! flips the image vertically
	void flipY() {
		#pragma omp parallel for
		for (int i = 0; i < (int)m_Width; i++) {
			for (int j = 0; j < (int)m_Height/2; j++) {
				T tmp = (*this)(j, i);
				(*this)(j, i) = (*this)((int)m_Height - j - 1, i);
				(*this)((int)m_Height - j - 1, i) = tmp;
			}
		}
	}

	//! flips the image horizontally
	void flipX() {
		#pragma omp parallel for
		for (int i = 0; i < (int)m_Width/2; i++) {
			for (int j = 0; j < (int)m_Height; j++) {
				T tmp = (*this)(j, i);
				(*this)(j, i) = (*this)(j, (int)m_Width - i - 1);
				(*this)(j, (int)m_Width - i - 1) = tmp;
			}
		}
	}


	//! returns the invalid value
	T getInvalidValue() const {
		return m_InvalidValue;
	}

	//! sets the invalid value
	void setInvalidValue(T invalidValue) {
		m_InvalidValue = invalidValue;
	}

	//! sets a pixel to the invalid value
	template <class S>
	void setInvalidValue(S y, S x) {
		setPixel(y, x, getInvalidValue());
	}

	//! returns true if a value is valid
	bool isValidValue(T value) const {
		return value != m_InvalidValue;
	}

	//! returns true if the depth value at position (x,y) is valid
	template <class S>
	bool isValidValue(S y, S x) const {
		return getPixel(y,x) != m_InvalidValue;
	}

	//! returns the number of channels per pixel (-1 if unknown)
	unsigned int getNumChannels() const  {
		if (std::is_same<T, double>::value || std::is_same<T, float >::value || std::is_same<T, UCHAR >::value ||	std::is_same<T, UINT  >::value || std::is_same<T, int   >::value) return 1;
		if (std::is_same<T, vec2d >::value || std::is_same<T, vec2f >::value || std::is_same<T, vec2uc>::value ||	std::is_same<T, vec2ui>::value || std::is_same<T, vec2i >::value) return 2;
		if (std::is_same<T, vec3d >::value || std::is_same<T, vec3f >::value || std::is_same<T, vec3uc>::value ||	std::is_same<T, vec3ui>::value || std::is_same<T, vec3i >::value) return 3;
		if (std::is_same<T, vec4d >::value || std::is_same<T, vec4f >::value || std::is_same<T, vec4uc>::value ||	std::is_same<T, vec4ui>::value || std::is_same<T, vec4i >::value) return 4;
		return -1;
	}

	//! returns the number of bits per channel (-1 if unknown);
	unsigned int getNumBytesPerChannel() const  {
		const unsigned int numChannels = getNumChannels();
		if (numChannels != -1) return sizeof(T)/numChannels;
		else return -1;
	}

	//! returns the storage requirsments per pixel
	unsigned int getNumBytesPerPixel() const {
		return sizeof(T);
	}

	//! computes the next mip map level of the image (box filtered image)
	void mipMap(BaseImage& result, bool ignoreInvalidPixels = false) const {
		result.allocateToSize(m_Height / 2, m_Width / 2);
		result.setInvalidValue(m_InvalidValue);

		if (!ignoreInvalidPixels) {
			#pragma omp parallel for
			for (int i = 0; (unsigned int)i < result.getHeight(); i++) {
				for (int j = 0; (unsigned int)j < result.getWidth(); j++) {
					result(i,j) = getPixel(2*i + 0, 2*j + 0) + getPixel(2*i + 1, 2*j + 0) + getPixel(2*i + 0, 2*j + 1) + getPixel(2*i + 1, 2*j + 1); 
					result(i,j) /= 4;
				}
			}
		} else {
			#pragma omp parallel for
			for (int i = 0; (unsigned int)i < result.getHeight(); i++) {
				for (int j = 0; (unsigned int)j < result.getWidth(); j++) {
					unsigned int valid = 0;
					T value = T();
					if (isValidValue(2*i + 0, 2*j + 0))	{
						valid++;
						value += getPixel(2*i + 0, 2*j + 0);
					}
					if (isValidValue(2*i + 1, 2*j + 0))	{
						valid++;
						value += getPixel(2*i + 1, 2*j + 0);
					}
					if (isValidValue(2*i + 0, 2*j + 1))	{
						valid++;
						value += getPixel(2*i + 0, 2*j + 1);
					}
					if (isValidValue(2*i + 1, 2*j + 1))	{
						valid++;
						value += getPixel(2*i + 1, 2*j + 1);
					}
					if (value == 0) {
						result(i,j) = result.getInvalidValue();
					} else {
						result(i,j) = value / valid;
					}
				}
			}
		}
	}

	//! nearest neighbor re-sampling
	void reSample(unsigned int newHeight, unsigned int newWidth) {
		if (m_Width != newWidth || m_Height != newHeight) {
			BaseImage res(newHeight, newWidth);
			for (unsigned int i = 0; i < newHeight; i++) {
				for (unsigned int j = 0; j < newWidth; j++) {
					const float y = (float)i/(newHeight-1);
					const float x = (float)j/(newWidth-1);
					res(i,j) = getPixel(y,x);
				}
			}
			*this = std::move(res);
		}
	}


	//! various operator overloads
	template<class U>
	void scale(const U& s) {
		for (unsigned int i = 0; i < m_Width*m_Height; i++) {
			m_Data[i] *= s;
		}
	}

	template<class U>
	BaseImage& operator*=(const U& s) {
		scale(s);
		return *this;
	}
	template<class U>
	BaseImage& operator/=(const U& s) {
		for (unsigned int i = 0; i < m_Width*m_Height; i++) {
			m_Data[i] /= s;
		}
		return *this;
	}
	template<class U>
	BaseImage& operator+=(const U& s) {
		for (unsigned int i = 0; i < m_Width*m_Height; i++) {
			m_Data[i] += s;
		}
		return *this;
	}
	template<class U>
	BaseImage& operator-=(const U& s) {
		for (unsigned int i = 0; i < m_Width*m_Height; i++) {
			m_Data[i] -= s;
		}
		return *this;
	}
protected:
	//! Allocates memory and sets the image size accordingly
	void create(unsigned int height, unsigned int width) {
		m_Height = height;
		m_Width = width;
		m_Data = new T[m_Width * m_Height];
	}

	//! Image data
	T* m_Data;

	//! Image width
	unsigned int m_Width;

	//! Image height
	unsigned int m_Height;

	//! Invalid image value
	T m_InvalidValue;

};

template<class BinaryDataBuffer, class BinaryDataCompressor, class T>
__forceinline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const BaseImage<T>& image) {
	s.writeData(image.getHeight());
	s.writeData(image.getWidth());
	s.writeData(image.m_InvalidValue);
	s.writeData((BYTE*)image.getDataPointer(), sizeof(T)*image.getWidth()*image.getHeight());
	return s;
}

template<class BinaryDataBuffer, class BinaryDataCompressor, class T>
__forceinline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, BaseImage<T>& image) {
	unsigned int height, width;
	s.readData(&height);
	s.readData(&width);
	s.readData(&image.m_InvalidValue);
	image.allocateToSize(height, width);
	s.readData((BYTE*)image.getDataPointer(), sizeof(T)*width*height);
	return s;
}


class DepthImage : public BaseImage<float> {
public:
	DepthImage() : BaseImage() {
		m_InvalidValue = -std::numeric_limits<float>::infinity();
	}

	DepthImage(unsigned int height, unsigned int width, const float *data) : BaseImage(height, width, data) {
		m_InvalidValue = -std::numeric_limits<float>::infinity();
	}
	DepthImage(unsigned int height, unsigned int width) : BaseImage(height, width) {
		m_InvalidValue = -std::numeric_limits<float>::infinity();
	}
	~DepthImage() {

	}

	//! Saves the depth image as a PPM file; note that there is a loss of precision
	void saveAsPPM( const std::string &filename ) const
	{
		std::ofstream out(filename, std::ofstream::out);

		if(out.fail())
		{
			std::cerr << "Error in function void __FUNCTION__ const: Can not open file " << filename << "!" << std::endl;
			exit(1);
		}

		out << "P3" << std::endl;
		out << "#" << filename << std::endl;
		out << m_Width << " " << m_Height << std::endl;

		out << "255" << std::endl;

		for (unsigned int i = 0; i < m_Height; i++)	{
			for (unsigned int j = 0; j < m_Width; j++)	{
				float res = getPixel(i,j);
				out <<	
					(int)convertValueToExternalPPMFormat(res) << " " <<
					(int)convertValueToExternalPPMFormat(res) << " " <<
					(int)convertValueToExternalPPMFormat(res) << " " << "\n";
			}
		}

		out.close();
	}

	void loadFromPPM(const std::string &filename) {
		std::ifstream file(filename);
		if (!file.is_open()) {
			throw std::ifstream::failure(std::string("Could not open file!").append(filename));
		}

		std::string s;
		getline(file, s); // Format
		getline(file, s); // Comment
		getline(file, s); // Width and Height


		unsigned int width, height;
		std::stringstream wh(s);
		wh >> width;
		wh >> height;
		allocateToSize(height, width);

		getline(file, s); // Max Value

		for (unsigned int i = 0; i < m_Height; i++) {
			for (unsigned int j = 0; j < m_Width; j++) {
				unsigned int c;
				vec3f color;

				file >> c; color.x = convertValueFromExternalPPMFormat((unsigned char)c);
				file >> c; color.y = convertValueFromExternalPPMFormat((unsigned char)c);
				file >> c; color.z = convertValueFromExternalPPMFormat((unsigned char)c);

				assert(c <= 255);
				assert(color.x == color.y && color.y == color.z);
				
				(*this)(i, j) = color.x;
			}
		}

		file.close();
	}


private:
	static unsigned char convertValueToExternalPPMFormat( float x ) 
	{
		if (x < (float)0) {
			std::cout << __FUNCTION__ << ": Warning value clamped!" << std::endl;
			return 0;
		}

		if (x > (float)1)	{
			std::cout << __FUNCTION__ << ": Warning value clamped!" << std::endl;
			return 255;
		}

		return (unsigned char)(x*(float)255.0 + (float)0.49999);
	}

	static float convertValueFromExternalPPMFormat( unsigned char x) 
	{
		return (float)x/255.0f;
	}
};


class ColorImageRGB : public BaseImage<vec3f> {
public:
	ColorImageRGB() : BaseImage() {
		m_InvalidValue = vec3f(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
	}

	ColorImageRGB(unsigned int height, unsigned int width, const vec3f *data) : BaseImage(height, width, data) {
		m_InvalidValue = vec3f(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
	}

	ColorImageRGB(unsigned int height, unsigned int width, const vec3uc *data, float scale = 255.0f) : BaseImage(height, width) {
		m_InvalidValue = vec3f(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());

#pragma omp parallel for
		for (int i = 0; i < (int)height; i++) {
			for (int j = 0; j < (int)width; j++) {
				vec3f value(	(float)data[i*width + j].x / scale,
								(float)data[i*width + j].y / scale,
								(float)data[i*width + j].z / scale
							);
				setPixel(i, j, value);
			}
		}
	}
	ColorImageRGB(unsigned int height, unsigned int width) : BaseImage(height, width) {
		m_InvalidValue = vec3f(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
	}

	ColorImageRGB(const DepthImage& depthImage) : BaseImage(depthImage.getHeight(), depthImage.getWidth()) {
		m_InvalidValue = vec3f(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());

		const float* data = depthImage.getDataPointer();
		float maxDepth = -FLT_MAX;
		float minDepth = +FLT_MAX;
		for (unsigned int i = 0; i < getWidth()*getHeight(); i++) {
			if (data[i] != depthImage.getInvalidValue()) {
				if (data[i] > maxDepth) maxDepth = data[i];
				if (data[i] < minDepth) minDepth = data[i];
			}
		}
		std::cout << "max Depth " << maxDepth << std::endl;
		std::cout << "min Depth " << minDepth << std::endl;
	
		for (unsigned int i = 0; i < getWidth()*getHeight(); i++) {
			if (data[i] != depthImage.getInvalidValue()) {
				m_Data[i] = BaseImageHelper::convertDepthToRGB(data[i], minDepth, maxDepth);
			} else {
				m_Data[i] = getInvalidValue();
			}
		}
	}
	~ColorImageRGB() {

	}

};


typedef ColorImageRGB PointImage;

typedef BaseImage<float>	ColorImageR32;
typedef BaseImage<vec3uc>	ColorImageR8G8B8;
typedef BaseImage<vec4uc>	ColorImageR8G8B8A8;

} // namespace ml


#endif

