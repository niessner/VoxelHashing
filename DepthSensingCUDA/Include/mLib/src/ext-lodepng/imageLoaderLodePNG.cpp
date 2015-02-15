
namespace ml
{

Bitmap LodePNG::load(const std::string &filename)
{
    if(!ml::util::fileExists(filename))
    {
        Console::log("LodePNG::load file not found: " + filename);
        return Bitmap();
    }
	Bitmap result;

	std::vector<BYTE> image;
	UINT width, height;

	UINT error = lodepng::decode(image, width, height, filename);

	MLIB_ASSERT_STR(!error, std::string(lodepng_error_text(error)) + ": " + filename);

	result.allocate(height, width);
	memcpy(result.ptr(), &image[0], 4 * width * height);

	return result;
}

void LodePNG::save(const Bitmap &bmp, const std::string &filename)
{
	const UINT pixelCount = bmp.rows() * bmp.cols();
	
	//
	// images should be saved with no transparency, which unfortunately requires us to make a copy of the bitmap data.
	//
	RGBColor *copy = new RGBColor[pixelCount];
	memcpy(copy, bmp.ptr(), pixelCount * 4);
	for(UINT i = 0; i < pixelCount; i++)
		copy[i].a = 255;

	lodepng::encode(filename, (const BYTE *)copy, bmp.cols(), bmp.rows(), LodePNGColorType::LCT_RGBA);
	delete[] copy;
}

}  // namespace ml
