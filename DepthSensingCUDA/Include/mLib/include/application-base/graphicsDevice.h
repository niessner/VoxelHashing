
#ifndef APPLICATION_BASE_GRAPHICSDEVICE_H_
#define APPLICATION_BASE_GRAPHICSDEVICE_H_

namespace ml {

enum GraphicsDeviceType
{
	GraphicsDeviceTypeD3D11,
};

class D3D11GraphicsDevice;
class GraphicsDevice
{
public:
	virtual ~GraphicsDevice() {}
	virtual void init(const WindowWin32 &window) = 0;
    virtual void resize(UINT width, UINT height) = 0;
	virtual void renderBeginFrame() = 0;
    virtual void clear(const vec4f &clearColor) = 0;
	virtual void renderEndFrame() = 0;

    void captureBackBuffer(Bitmap &result)
    {
        captureBackBufferInternal(result);
    }

    Bitmap captureBackBuffer()
    {
        Bitmap result;
        captureBackBuffer(result);
        return result;
    }

	GraphicsDeviceType type() const
	{
		return m_type;
	}

	D3D11GraphicsDevice& castD3D11() const
	{
		return *((D3D11GraphicsDevice*)this);
	}

protected:
    virtual void captureBackBufferInternal(Bitmap &result) = 0;
	GraphicsDeviceType m_type;
};

}  // namespace ml

#endif  // APPLICATION_BASE_GRAPHICSDEVICE_H_
