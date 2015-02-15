
#ifndef CORE_MATH_MATHUTIL_H_
#define CORE_MATH_MATHUTIL_H_

namespace ml {
namespace math {

//
// returns the <axis, eigenvalue> pairs for the PCA of the given 3D points.
//
template <class T>
vector< std::pair<point3d<T>, T> > pointSetPCA(const std::vector< point3d<T> > &points)
{
    auto mean = std::accumulate(points.begin(), points.end(), point3d<T>::origin) / (T)points.size();

    DenseMatrix<T> covariance(3, 3, (T)0.0);
    
    for (const auto &p : points)
    {
        auto recenteredPt = p - mean;
        auto tensor = Matrix3x3<T>::tensorProduct(recenteredPt, recenteredPt);
        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 3; x++)
                covariance(y, x) += tensor(y, x);
    }

    covariance /= (T)(points.size() - 1);

    auto system = covariance.eigenSystem();
    const auto &v = system.eigenvectors;
    
    vector< std::pair<point3d<T>, T> > result;
    result.push_back(std::make_pair(point3d<T>(v(0, 0), v(0, 1), v(0, 2)), system.eigenvalues[0]));
    result.push_back(std::make_pair(point3d<T>(v(1, 0), v(1, 1), v(1, 2)), system.eigenvalues[1]));
    result.push_back(std::make_pair(point3d<T>(v(2, 0), v(2, 1), v(2, 2)), system.eigenvalues[2]));
    return result;
}

//
// returns the <axis, eigenvalue> pairs for the PCA of the given 2D points.
//
template <class T>
vector< std::pair<point2d<T>, T> > pointSetPCA(const std::vector< point2d<T> > &points)
{
    auto mean = std::accumulate(points.begin(), points.end(), point2d<T>::origin) / (T)points.size();

    DenseMatrix<T> covariance(2, 2, (T)0.0);

    for (const auto &p : points)
    {
        auto recenteredPt = p - mean;
        auto tensor = Matrix2x2<T>::tensorProduct(recenteredPt, recenteredPt);
        for (int y = 0; y < 2; y++)
            for (int x = 0; x < 2; x++)
                covariance(y, x) += tensor(y, x);
    }

    /*DenseMatrix<T> B(2, (UINT)points.size());
    for (UINT pointIndex = 0; pointIndex < (UINT)points.size(); pointIndex++)
    {
        B(0, pointIndex) = points[pointIndex][0] - mean[0];
        B(1, pointIndex) = points[pointIndex][1] - mean[1];
    }
    auto covariance = B * B.transpose();*/

    covariance /= (T)(points.size() - 1);

    auto system = covariance.eigenSystem();
    const auto &v = system.eigenvectors;

    vector< std::pair<point2d<T>, T> > result;
    result.push_back(std::make_pair(point2d<T>(v(0, 0), v(1, 0)), system.eigenvalues[0]));
    result.push_back(std::make_pair(point2d<T>(v(0, 1), v(1, 1)), system.eigenvalues[1]));
    return result;
}


}
}  // namespace ml

#endif  // CORE_MATH_SPARSEMATRIX_H_
