#ifndef _COREMESH_TRIMESHSAMPLER_H_
#define _COREMESH_TRIMESHSAMPLER_H_

namespace ml {

template<class T>
class TriMeshSampler
{
public:
    struct Sample
    {
        vec3f pos;
        vec3f normal;

        UINT meshIndex;
        UINT triangleIndex;
        vec2f uv;
    };

    typedef std::pair<const TriMesh<T>*, mat4f> MeshData;

    //
    // sampleDensity is the desired number of samples per square meter of mesh surface area. maxSampleCount is an absolute cutoff.
    // The normalPredicate function determines whether a triangle with the given normal is acceptable for sampling.
    //

    static std::vector<Sample> sample(const std::vector< std::pair<const TriMesh<T>*, mat4f> > &meshes, float sampleDensity, UINT maxSampleCount, const vec3f &normal)
    {
        return sample(meshes, sampleDensity, maxSampleCount, [&](const vec3f &n) { return vec3f::distSq(normal, n) < 1e-3f; });
    }

    static std::vector<Sample> sample(const std::vector< std::pair<const TriMesh<T>*, mat4f> > &meshes, float sampleDensity, UINT maxSampleCount, const std::function<bool(const vec3f&)> &normalPredicate);

private:
    static double directionalSurfaceArea(const MeshData &mesh, const std::function<bool(const vec3f&)> &normalPredicate);

    static double triangleArea(const MeshData &mesh, UINT triangleIndex)
    {
        return triangleArea(mesh, mesh.first->getIndices()[triangleIndex]);
    }
    static point3d<T> triangleNormal(const MeshData &mesh, UINT triangleIndex)
    {
        return triangleNormal(mesh, mesh.first->getIndices()[triangleIndex]);
    }

    static double triangleArea(const MeshData &mesh, const vec3ui &tri);
    static point3d<T> triangleNormal(const MeshData &mesh, const vec3ui &tri);
    static Sample sampleTriangle(const MeshData &mesh, UINT meshIndex, UINT triangleIndex, double sampleValue);
    static vec2f stratifiedSample2D(double s, UINT depth = 0);
};

template<class T>
std::vector<typename TriMeshSampler<T>::Sample> TriMeshSampler<T>::sample(const std::vector< std::pair<const TriMesh<T>*, mat4f> > &meshes, float sampleDensity, UINT maxSampleCount, const std::function<bool(const vec3f&)> &normalPredicate)
{
    double totalArea = 0.0;
    for (const auto &m : meshes)
        totalArea += directionalSurfaceArea(m, normalPredicate);
    double areaScale = 1.0 / totalArea;

    if (totalArea == 0.0)
    {
        return std::vector<Sample>();
    }

    // Indices of mesh index <-> triangle index pairs with acceptable normals
    std::vector< std::pair<UINT, UINT> > meshTriangleIndices;

    UINT meshIndex = 0;
    for (auto &m : meshes)
    {
        const auto &indices = m.first->getIndices();
        for (UINT triangleIndex = 0; triangleIndex < indices.size(); triangleIndex++)
        {
            if (normalPredicate(triangleNormal(m, indices[triangleIndex])))
                meshTriangleIndices.push_back(std::make_pair(meshIndex, triangleIndex));
        }
        meshIndex++;
    }

    const UINT sampleCount = std::min(maxSampleCount, UINT(totalArea * sampleDensity));

    std::vector<Sample> samples(sampleCount);

    auto activeMeshTriangle = meshTriangleIndices.begin();
    double samplingTriangleAreaRatio = triangleArea(meshes[activeMeshTriangle->first], activeMeshTriangle->second) * areaScale;
    double accumulatedAreaRatio = samplingTriangleAreaRatio;

    for (UINT sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
        double intervalMin = double(sampleIndex) / double(sampleCount);
        double intervalMax = double(sampleIndex + 1) / double(sampleCount);
        double sampleValue = util::randomUniform(intervalMin, intervalMax);

        while (accumulatedAreaRatio < sampleValue && activeMeshTriangle != meshTriangleIndices.end())
        {
            activeMeshTriangle++;
            samplingTriangleAreaRatio = triangleArea(meshes[activeMeshTriangle->first], activeMeshTriangle->second) * areaScale;
            accumulatedAreaRatio += samplingTriangleAreaRatio;
        }

        double triangleValue = math::clamp(math::linearMap(accumulatedAreaRatio - samplingTriangleAreaRatio, accumulatedAreaRatio, 0.0, 1.0, sampleValue), 0.0, 1.0);

        samples[sampleIndex] = sampleTriangle(meshes[activeMeshTriangle->first], activeMeshTriangle->first, activeMeshTriangle->second, triangleValue);
    }

    return samples;
}

template<class T>
double TriMeshSampler<T>::directionalSurfaceArea(const MeshData &mesh, const std::function<bool(const vec3f&)> &normalPredicate)
{
    double result = 0.0;
    for (const vec3ui &tri : mesh.first->getIndices())
    {
        if (normalPredicate(triangleNormal(mesh, tri)))
            result += triangleArea(mesh, tri);
    }
    return result;
}

template<class T>
double TriMeshSampler<T>::triangleArea(const MeshData &mesh, const vec3ui &tri)
{
    ml::point3d<T> v[3];
    for (int i = 0; i < 3; i++)
        v[i] = mesh.second.transformAffine(mesh.first->getVertices()[tri[i]].position);
    return math::triangleArea(v[0], v[1], v[2]);
}

template<class T>
point3d<T> TriMeshSampler<T>::triangleNormal(const MeshData &mesh, const vec3ui &tri)
{
    ml::point3d<T> v[3];
    for (int i = 0; i < 3; i++)
        v[i] = mesh.second.transformAffine(mesh.first->getVertices()[tri[i]].position);
    return math::triangleNormal(v[0], v[1], v[2]);
}

template<class T>
typename TriMeshSampler<T>::Sample TriMeshSampler<T>::sampleTriangle(const MeshData &mesh, UINT meshIndex, UINT triangleIndex, double sampleValue)
{
    vec3ui tri = mesh.first->getIndices()[triangleIndex];

    point3d<T> v[3];
    for (int i = 0; i < 3; i++)
        v[i] = mesh.second.transformAffine(mesh.first->getVertices()[tri[i]].position);

    vec2f uv = stratifiedSample2D(sampleValue);
    if (uv.x + uv.y > 1.0f)
    {
        uv = vec2d(1.0 - uv.y, 1.0 - uv.x);
    }

    Sample result;
    result.pos = v[0] + (v[1] - v[0]) * uv.x + (v[2] - v[0]) * uv.y;
    result.normal = math::triangleNormal(v[0], v[1], v[2]);
    result.meshIndex = meshIndex;
    result.triangleIndex = triangleIndex;
    result.uv = uv;
    return result;
}

template<class T>
vec2f TriMeshSampler<T>::stratifiedSample2D(double s, UINT depth)
{
    if (depth == 10)
    {
        return vec2f((float)util::randomUniform(0.0f, 1.0f), (float)util::randomUniform(0.0f, 1.0f));
    }

    vec2d basePoint;
    double baseValue;
    if (s < 0.25)
    {
        baseValue = 0.0;
        basePoint = vec2f(0.0f, 0.0f);
    }
    else if (s < 0.5)
    {
        baseValue = 0.25;
        basePoint = vec2f(0.5f, 0.0f);
    }
    else if (s < 0.75)
    {
        baseValue = 0.5;
        basePoint = vec2f(0.0f, 0.5f);
    }
    else
    {
        baseValue = 0.75;
        basePoint = vec2f(0.5f, 0.5f);
    }

    return basePoint + stratifiedSample2D((s - baseValue) * 4.0, depth + 1) * 0.5f;
}

} // ml

// below is code for my pre-C++-11 MeshSampler
// this can be deleted once we get a functional sampler.
/*
class MeshSampler
{
public:
    static void Sample(const BaseMesh &m, UINT sampleCount, Vector<MeshSample> &samples);
    static void Sample(const Vector< pair<const BaseMesh *, Matrix4> > &geometry, UINT sampleCount, Vector<MeshSample> &samples);

    static void Sample(const BaseMesh &m, UINT& sampleCount, float densityThresh, const Vec3f &direction, float targetOrientation, float orientThresh, Vector<MeshUVSample> &samples);
    static void Sample(const Vector< pair<const BaseMesh *, Matrix4> > &geometry, UINT sampleCount, float densityThresh, const Vec3f &direction, float targetOrientation, float orientThresh, Vector<MeshUVSample> &samples);

private:
    static double DirectionalSurfaceArea(const BaseMesh &m, const Vec3f &direction, float targetOrientation, float orientThresh);
    static float GetTriangleArea(const BaseMesh &m, UINT triangleIndex);
    static Vec3f GetTriangleNormal(const BaseMesh &m, UINT triangleIndex);
    static MeshSample SampleTriangle(const BaseMesh &m, UINT triangleIndex, double sampleValue);
    static MeshUVSample SampleTriangleUV(const BaseMesh &m, UINT triangleIndex, double sampleValue);
    static Vec2f StratifiedSample2D(double s, UINT depth = 0);
};

double MeshSampler::DirectionalSurfaceArea(const BaseMesh &m, const Vec3f &direction, float targetOrientation, float orientThresh)
{
    const UINT TriangleCount = m.FaceCount();
    const DWORD *MyIndices = m.Indices();
    const MeshVertex *MyVertices = m.Vertices();
    double Result = 0.0;
    for (UINT TriangleIndex = 0; TriangleIndex < TriangleCount; TriangleIndex++)
    {
        Vec3f V[3];
        for (UINT LocalVertexIndex = 0; LocalVertexIndex < 3; LocalVertexIndex++)
        {
            V[LocalVertexIndex] = MyVertices[MyIndices[TriangleIndex * 3 + LocalVertexIndex]].Pos;
        }
        Vec3f normal = Math::TriangleNormal(V[0], V[1], V[2]);
        if (fabs(Vec3f::Dot(normal, direction) - targetOrientation) <= orientThresh)
            Result += Math::TriangleArea(V[0], V[1], V[2]);
    }
    return Result;
}

void MeshSampler::Sample(const BaseMesh &m, UINT sampleCount, Vector<MeshSample> &samples)
{
    double totalArea = m.SurfaceArea();
    double areaScale = 1.0 / totalArea;
    if (totalArea == 0.0)
    {
        samples.FreeMemory();
        return;
    }

    samples.Allocate(sampleCount);

    UINT samplingTriangleIndex = 0;
    double samplingTriangleAreaRatio = GetTriangleArea(m, samplingTriangleIndex) * areaScale;
    double accumulatedAreaRatio = samplingTriangleAreaRatio;

    for (UINT sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
        double intervalMin = double(sampleIndex) / double(sampleCount);
        double intervalMax = double(sampleIndex + 1) / double(sampleCount);
        double sampleValue = intervalMin + rnd() * (intervalMax - intervalMin);

        while (accumulatedAreaRatio < sampleValue && samplingTriangleIndex < m.FaceCount() - 1)
        {
            samplingTriangleIndex++;
            samplingTriangleAreaRatio = GetTriangleArea(m, samplingTriangleIndex) * areaScale;
            accumulatedAreaRatio += samplingTriangleAreaRatio;
        }

        double triangleValue = Utility::Bound(Math::LinearMap(accumulatedAreaRatio - samplingTriangleAreaRatio, accumulatedAreaRatio, 0.0, 1.0, sampleValue), 0.0, 1.0);
        samples[sampleIndex] = SampleTriangle(m, samplingTriangleIndex, triangleValue);
    }
}

void MeshSampler::Sample(const BaseMesh &m, UINT& sampleCount, float densityThresh, const Vec3f &direction, float targetOrientation, float orientThresh, Vector<MeshUVSample> &samples)
{
    double totalArea = DirectionalSurfaceArea(m, direction, targetOrientation, orientThresh);
    double areaScale = 1.0 / totalArea;
    if (totalArea == 0.0)
    {
        sampleCount = 0;
        samples.FreeMemory();
        return;
    }

    // Threshold sample count according to maximum density criterion
    sampleCount = Math::Min(sampleCount, UINT(totalArea / densityThresh));

    samples.Allocate(sampleCount);

    // Indices of triangles with good normals
    Vector<UINT> goodTriIndices;
    for (UINT i = 0; i < m.FaceCount(); i++)
    {
        Vec3f normal = GetTriangleNormal(m, i);
        if (fabs(Vec3f::Dot(normal, Vec3f::eZ) - targetOrientation) <= orientThresh)
            goodTriIndices.PushEnd(i);
    }

    UINT indexIntoGoodTris = 0;
    UINT samplingTriangleIndex = goodTriIndices[indexIntoGoodTris];
    double samplingTriangleAreaRatio = GetTriangleArea(m, samplingTriangleIndex) * areaScale;
    double samplingTriangleOrientation = Vec3f::Dot(GetTriangleNormal(m, samplingTriangleIndex), Vec3f::eZ);
    double accumulatedAreaRatio = samplingTriangleAreaRatio;

    for (UINT sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
        double intervalMin = double(sampleIndex) / double(sampleCount);
        double intervalMax = double(sampleIndex + 1) / double(sampleCount);
        double sampleValue = intervalMin + rnd() * (intervalMax - intervalMin);

        while (accumulatedAreaRatio < sampleValue && indexIntoGoodTris < goodTriIndices.Length() - 1)
        {
            indexIntoGoodTris++;
            samplingTriangleIndex = goodTriIndices[indexIntoGoodTris];
            samplingTriangleAreaRatio = GetTriangleArea(m, samplingTriangleIndex) * areaScale;
            samplingTriangleOrientation = Vec3f::Dot(GetTriangleNormal(m, samplingTriangleIndex), Vec3f::eZ);
            accumulatedAreaRatio += samplingTriangleAreaRatio;
        }

        double triangleValue = Utility::Bound(Math::LinearMap(accumulatedAreaRatio - samplingTriangleAreaRatio, accumulatedAreaRatio, 0.0, 1.0, sampleValue), 0.0, 1.0);

        MeshUVSample &curSample = samples[sampleIndex];
        curSample = SampleTriangleUV(m, samplingTriangleIndex, triangleValue);
        curSample.triangleIndex = samplingTriangleIndex;
        curSample.meshIndex = 0;
    }
}

void MeshSampler::Sample(const Vector< pair<const BaseMesh *, Matrix4> > &meshList, UINT sampleCount, Vector<MeshSample> &samples)
{
    Mesh allMeshes;
    allMeshes.LoadMeshList(meshList);
    Sample(allMeshes, sampleCount, samples);
}

void MeshSampler::Sample(const Vector< pair<const BaseMesh *, Matrix4> > &meshList, UINT sampleCount, float densityThresh, const Vec3f &direction, float targetOrientation, float orientThresh, Vector<MeshUVSample> &samples)
{
    Mesh allMeshes;
    allMeshes.LoadMeshList(meshList);
    Sample(allMeshes, sampleCount, densityThresh, direction, targetOrientation, orientThresh, samples);

    Vector<UINT> triangleToMeshIndex(allMeshes.FaceCount());
    Vector<UINT> meshIndexToBaseTriangleIndex(meshList.Length());
    UINT curIndex = 0;
    for (UINT meshIndex = 0; meshIndex < meshList.Length(); meshIndex++)
    {
        meshIndexToBaseTriangleIndex[meshIndex] = curIndex;
        if (meshList[meshIndex].first != nullptr)
        {
            const UINT triangleCount = meshList[meshIndex].first->FaceCount();
            for (UINT triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++)
            {
                triangleToMeshIndex[curIndex++] = meshIndex;
            }
        }
    }

    for (UINT sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
        MeshUVSample &curSample = samples[sampleIndex];
        curSample.meshIndex = triangleToMeshIndex[curSample.triangleIndex];
        curSample.triangleIndex = curSample.triangleIndex - meshIndexToBaseTriangleIndex[curSample.meshIndex];
    }
}

float MeshSampler::GetTriangleArea(const BaseMesh &m, UINT triangleIndex)
{
    const DWORD *indices = m.Indices();
    const MeshVertex *vertices = m.Vertices();

    Vec3f V[3];
    for (UINT vertexIndex = 0; vertexIndex < 3; vertexIndex++)
    {
        V[vertexIndex] = vertices[indices[triangleIndex * 3 + vertexIndex]].Pos;
    }
    return Math::TriangleArea(V[0], V[1], V[2]);
}

Vec3f MeshSampler::GetTriangleNormal(const BaseMesh &m, UINT triangleIndex)
{
    const DWORD *indices = m.Indices();
    const MeshVertex *vertices = m.Vertices();

    Vec3f V[3];
    for (UINT vertexIndex = 0; vertexIndex < 3; vertexIndex++)
    {
        V[vertexIndex] = vertices[indices[triangleIndex * 3 + vertexIndex]].Pos;
    }
    return Math::TriangleNormal(V[0], V[1], V[2]);
}

MeshSample MeshSampler::SampleTriangle(const BaseMesh &m, UINT triangleIndex, double sampleValue)
{
    const DWORD *indices = m.Indices();
    const MeshVertex *vertices = m.Vertices();

    Vec3f V[3];
    for (UINT vertexIndex = 0; vertexIndex < 3; vertexIndex++)
    {
        V[vertexIndex] = vertices[indices[triangleIndex * 3 + vertexIndex]].Pos;
    }

    MeshSample result;

    result.normal = Math::TriangleNormal(V[0], V[1], V[2]);

    Vec2f uv = StratifiedSample2D(sampleValue);
    if (uv.x + uv.y > 1.0f)
    {
        uv = Vec2f(1.0f - uv.y, 1.0f - uv.x);
    }

    result.pos = V[0] + (V[1] - V[0]) * uv.x + (V[2] - V[0]) * uv.y;

    return result;
}

MeshUVSample MeshSampler::SampleTriangleUV(const BaseMesh &m, UINT triangleIndex, double sampleValue)
{
    const DWORD *indices = m.Indices();
    const MeshVertex *vertices = m.Vertices();

    Vec3f V[3];
    for (UINT vertexIndex = 0; vertexIndex < 3; vertexIndex++)
    {
        V[vertexIndex] = vertices[indices[triangleIndex * 3 + vertexIndex]].Pos;
    }

    Vec2f uv = StratifiedSample2D(sampleValue);
    if (uv.x + uv.y > 1.0f)
    {
        uv = Vec2f(1.0f - uv.y, 1.0f - uv.x);
    }

    MeshUVSample result;
    result.uv = uv;
    result.pos = V[0] + (V[1] - V[0]) * uv.x + (V[2] - V[0]) * uv.y;
    result.normal = Math::TriangleNormal(V[0], V[1], V[2]);

    return result;
}

Vec2f MeshSampler::StratifiedSample2D(double s, UINT depth)
{
    if (depth == 10)
    {
        return Vec2f(rnd(), rnd());
    }

    Vec2f basePoint;
    double baseValue;
    if (s < 0.25)
    {
        baseValue = 0.0;
        basePoint = Vec2f(0.0f, 0.0f);
    }
    else if (s < 0.5)
    {
        baseValue = 0.25;
        basePoint = Vec2f(0.5f, 0.0f);
    }
    else if (s < 0.75)
    {
        baseValue = 0.5;
        basePoint = Vec2f(0.0f, 0.5f);
    }
    else
    {
        baseValue = 0.75;
        basePoint = Vec2f(0.5f, 0.5f);
    }

    return basePoint + StratifiedSample2D((s - baseValue) * 4.0, depth + 1) * 0.5;
}
*/

#endif