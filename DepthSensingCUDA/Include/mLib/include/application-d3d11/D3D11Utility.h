
#ifndef APPLICATION_D3D11_D3D11UTILITY_H_
#define APPLICATION_D3D11_D3D11UTILITY_H_

namespace ml {

class D3D11Utility
{
public:
	static ID3DBlob* CompileShader(const std::string &filename, const std::string &entryPoint, const std::string &shaderModel);
};

}  // namespace ml

#endif  // APPLICATION_D3D11_D3D11UTILITY_H_
