#ifndef _EXT_LODEPNG_IMAGELOADERLODEPNG_H_
#define _EXT_LODEPNG_IMAGELOADERLODEPNG_H_

namespace ml {

class LodePNG
{
public:
	static Bitmap load(const std::string &filename);
	static void save(const Bitmap &bmp, const std::string &filename);
};

}  // namespace ml

#endif  // _EXT_LODEPNG_IMAGELOADERLODEPNG_H_
