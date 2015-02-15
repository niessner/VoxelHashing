#pragma once

#ifndef CORE_UTIL_COLORGRADIENT_H_
#define CORE_UTIL_COLORGRADIENT_H_

namespace ml {

class ColorGradient
{
public:
    ColorGradient() {}
    ColorGradient(const Bitmap &bmp, RGBColor leftColor = RGBColor::Black, RGBColor rightColor = RGBColor::White)
    {
        m_colors = bmp.getRow(0);
    }

    RGBColor value(double x) const
    {
        if(x < 0.0) return m_leftColor;
        if(x > 1.0) return m_rightColor;
        double s = x * m_colors.size();
        int f = (int)floor(s);
        RGBColor a = m_colors[(UINT)math::clamp(f, 0, (int)m_colors.size() - 1)];
        RGBColor b = m_colors[(UINT)math::clamp(f + 1, 0, (int)m_colors.size() - 1)];
        return RGBColor::interpolate(a, b, (float)(s - f));
    }

private:
    std::vector<RGBColor> m_colors;
    RGBColor m_leftColor, m_rightColor;
};

} // namespace ml


#endif // CORE_UTIL_COLORGRADIENT_H_