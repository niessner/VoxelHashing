#ifndef CORE_GRAPHICS_RGBCOLOR_H_
#define CORE_GRAPHICS_RGBCOLOR_H_

namespace ml
{

struct RGBColor
{
    RGBColor() {}
    RGBColor(BYTE _r, BYTE _g, BYTE _b)
	{
		r = _r;
		g = _g;
		b = _b;
		a = 255;
	}
    RGBColor(BYTE _r, BYTE _g, BYTE _b, BYTE _a)
	{
		r = _r;
		g = _g;
		b = _b;
		a = _a;
	}
	
	RGBColor(const std::string &hex);
    explicit RGBColor(const vec3f &v);
    explicit RGBColor(const vec4f &v);

    RGBColor flipBlueAndRed() const
    {
        return RGBColor(b, g, r, a);
    }

    RGBColor grayscale() const
    {
        BYTE avg = BYTE(((int)r + (int)g + (int)b) / 3);
        return RGBColor(avg, avg, avg, 255);
    }

    RGBColor inverse() const
    {
        return RGBColor(255 - r, 255 - g, 255 - b, 255 - a);
    }

	static UINT distL1(RGBColor a, RGBColor b)
	{
		return std::abs(int(a.r) - int(b.r)) +
			std::abs(int(a.g) - int(b.g)) +
			std::abs(int(a.b) - int(b.b));
	}

	static UINT distL2(RGBColor a, RGBColor b)
	{
		int DiffR = int(a.r) - int(b.r);
		int DiffG = int(a.g) - int(b.g);
		int DiffB = int(a.b) - int(b.b);
		return DiffR * DiffR + DiffG * DiffG + DiffB * DiffB;
	}

	static RGBColor randomColor()
	{
		return RGBColor(rand() & 255, rand() & 255, rand() & 255);
	}

    static RGBColor interpolate(RGBColor LowColor, RGBColor HighColor, float s);

	operator vec3f() const
	{
		const float scale = 1.0f / 255.0f;
		return vec3f(r * scale, g * scale, b * scale);
	}
    operator vec4f() const
    {
        const float scale = 1.0f / 255.0f;
        return vec4f(r * scale, g * scale, b * scale, a * scale);
    }

    static const RGBColor White;
    static const RGBColor Red;
    static const RGBColor Green;
    static const RGBColor Gray;
    static const RGBColor Blue;
    static const RGBColor Yellow;
    static const RGBColor Orange;
    static const RGBColor Magenta;
    static const RGBColor Black;
    static const RGBColor Cyan;
    static const RGBColor Purple;

    BYTE r, g, b, a;
};

//
// Exact comparison functions.  Does not match the alpha channel.
//
inline bool operator == (RGBColor left, RGBColor right)
{
    return ((left.r == right.r) && (left.g == right.g) && (left.b == right.b));
}

inline bool operator != (RGBColor left, RGBColor right)
{
    return ((left.r != right.r) || (left.g != right.g) || (left.b != right.b));
}

typedef Grid2<RGBColor> Bitmap;

}  // namespace ml

#endif  // CORE_GRAPHICS_RGBCOLOR_H_
