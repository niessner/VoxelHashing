
#ifndef APPLICATION_BASE_GRAPHICSASSET_H_
#define APPLICATION_BASE_GRAPHICSASSET_H_

namespace ml {

enum GraphicsAssetType
{
    GraphicsAssetGeneric,
    GraphicsAssetTexture,
};

class GraphicsDevice;
class GraphicsAsset
{
public:
	virtual void release(GraphicsDevice &g) = 0;
	virtual void reset(GraphicsDevice &g) = 0;
    virtual GraphicsAssetType type() const
    {
        return GraphicsAssetGeneric;
    }
};

}  // namespace ml

#endif  // APPLICATION_BASE_GRAPHICSASSET_H_