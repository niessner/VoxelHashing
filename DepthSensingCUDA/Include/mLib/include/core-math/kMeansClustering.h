
#ifndef CORE_MATH_KMEANSCLUSTERING_H_
#define CORE_MATH_KMEANSCLUSTERING_H_

namespace ml {

class vec2fKMeansMetric
{
public:
	static float Dist(const vec2f &L, const vec2f &R)
	{
		return vec2f::distSq(L, R);
	}
};

class vec3fKMeansMetric
{
public:
	static float Dist(const vec3f &L, const vec3f &R)
	{
		return vec3f::distSq(L, R);
	}
};

template<class T>
class MathVectorKMeansMetric
{
public:
    static float Dist(const MathVector<T> &L, const MathVector<T> &R)
    {
        float sum = 0.0f;
        for(UINT i = 0; i < L.size(); i++)
        {
            T diff = L[i] - R[i];
            sum += (float)(diff * diff);
        }
        return sum;
    }
};

template<class T>
struct KMeansCluster
{
public:
	void init(const T &start)
	{
		center = start;
	}
	void finalizeIteration(const T &fallbackElement)
	{
		if(entries.size() == 0)
		{
			center = fallbackElement;
			return;
		}

		float sum = entries[0].second;
		T newCenter = entries[0].first;

		for(UINT entryIndex = 1; entryIndex < entries.size(); entryIndex++)
		{
			sum += entries[entryIndex].second;
			newCenter += entries[entryIndex].first;
		}

		center = newCenter;
		if(sum > 0.0f)
			center *= (1.0f / sum);
		entries.clear();
	}
	void addEntry(const T &preWeightedEntry, float weight)
	{
        entries.push_back(std::pair<T, float>(preWeightedEntry, weight));
	}
	void addEntry(const T &entry)
	{
		entries.push_back( std::pair<T, float>(entry, 1.0f) );
	}

	T center;
	std::vector< std::pair<T, float> > entries;
};

template<class T, class Metric>
class KMeansClustering
{
public:
	void cluster(const std::vector<T> &elements, UINT clusterCount, UINT maxIterations = 0, bool verbose = true, double maxDelta = 0.0)
	{
		if(verbose) Console::log(std::string("k-means clustering, ") + std::to_string(elements.size()) + std::string(" points, ") + std::to_string(clusterCount) + std::string(" clusters"));

		MLIB_ASSERT_STR(elements.size() >= clusterCount, "Too many clusters");

		m_clusters.resize(clusterCount);
		for(UINT clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++)
			m_clusters[clusterIndex].init(elements[rand() % elements.size()]);

		std::vector<UINT> storage(elements.size());

		UINT pass = 0;
		std::vector<T> previousClusterCenters(m_clusters.size());
		bool converged = false;
		while(!converged)
		{
			pass++;
			for(UINT clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++) {
				previousClusterCenters[clusterIndex] = m_clusters[clusterIndex].center;
			}
			iterate(elements, storage);

			double delta = 0.0;
			for(UINT clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++)
				delta += Metric::Dist(previousClusterCenters[clusterIndex], m_clusters[clusterIndex].center);

			converged = (delta <= maxDelta || pass == maxIterations);
			if(verbose) Console::log(std::string("pass ") + std::to_string(pass) + std::string(", ") + std::string("delta=") + std::to_string(delta));
		}
	}

	void cluster(const std::vector<T> &elements, const std::vector<float> &weights, UINT clusterCount, UINT maxIterations = 0, bool verbose = true, double maxDelta = 0.0)
	{
		if(verbose) Console::log(std::string("k-means clustering, ") + std::to_string(elements.size()) + std::string(" points, ") + std::to_string(clusterCount) + std::string(" clusters"));

		MLIB_ASSERT_STR(elements.size() >= clusterCount, "Too many clusters");
		MLIB_ASSERT_STR(elements.size() == weights.size(), "Incorrect number of weights");

		m_clusters.resize(clusterCount);
		for(UINT clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++)
            m_clusters[clusterIndex].init(elements[rand() % elements.size()]);

		std::vector<T> weightedElements = elements;
		for(UINT elementIndex = 0; elementIndex < elements.size(); elementIndex++)
			weightedElements[elementIndex] *= weights[elementIndex];

		std::vector<UINT> storage(elements.size());

		UINT pass = 0;
		std::vector<T> previousClusterCenters(m_clusters.size());
		bool converged = false;
		while(!converged)
		{
			pass++;
			for(UINT clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++)
				previousClusterCenters[clusterIndex] = m_clusters[clusterIndex].center;

			iterate(elements, weightedElements, weights, storage);

			double delta = 0.0;
			for(UINT clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++)
				delta += Metric::Dist(previousClusterCenters[clusterIndex], m_clusters[clusterIndex].center);

			converged = (delta <= maxDelta || pass == maxIterations);
			if(verbose) Console::log(std::string("pass ") + std::to_string(pass) + std::string(", ") + std::string("delta=") + std::to_string(delta));
		}
	}

	const T& clusterCenter(UINT clusterIndex) const
	{
		return m_clusters[clusterIndex].center;
	}

    size_t findBestClusterRepresenative(const std::vector<T> &elements, UINT clusterIndex) const
    {
        size_t bestIndex = 0;
        float bestDist = std::numeric_limits<float>::max();
        for (UINT elementIndex = 0; elementIndex < elements.size(); elementIndex++)
        {
            float curDist = Metric::Dist(m_clusters[clusterIndex].center, elements[elementIndex]);
            if (curDist < bestDist)
            {
                bestDist = curDist;
                bestIndex = elementIndex;
            }
        }
        return bestIndex;
    }

    float quantizationError(const std::vector<T> &elements) const
    {
        float sum = 0.0f;
        for(const T &e : elements)
        {
            sum += Metric::Dist(quantizeToNearestClusterCenter(e), e);
        }
        return sum / elements.size();
    }

	const T& quantizeToNearestClusterCenter(const T &element) const
	{
		return m_clusters[quantizeToNearestClusterIndex(element)].center;
	}

	UINT quantizeToNearestClusterIndex(const T &element) const
	{
		UINT closestClusterIndex = 0;
		double closestClusterDist = Metric::Dist(element, m_clusters[0].center);
		for(UINT clusterIndex = 1; clusterIndex < m_clusters.size(); clusterIndex++)
		{
			double curClusterDist = Metric::Dist(element, m_clusters[clusterIndex].center);
			if(curClusterDist < closestClusterDist)
			{
				closestClusterIndex = clusterIndex;
				closestClusterDist = curClusterDist;
			}
		}
		return closestClusterIndex;
	}

    std::vector< std::pair<UINT, float> > rankClusterDistances(const T &element) const
    {
        std::vector< std::pair<UINT, float> > result;
        for(UINT clusterIndex = 0; clusterIndex < m_clusters.size(); clusterIndex++)
            result.push_back( std::make_pair(clusterIndex, Metric::Dist(element, m_clusters[clusterIndex].center) ) );
        std::sort(result.begin(), result.end(), [](const std::pair<UINT, float> &a, const std::pair<UINT, float> &b) { return a.second < b.second; } );
        return result;
    }

	UINT clusterCount() const
	{
		return (UINT)m_clusters.size();
	}

private:

	void iterate(const std::vector<T> &elements, std::vector<UINT> &storage)
	{
		const int elementCount = (int)elements.size();
		const T* elementPtr = &elements[0];
		const int clusterCount = (int)m_clusters.size();
		KMeansCluster<T> *clustersPtr = &m_clusters[0];

#ifdef MLIB_OPENMP
#pragma omp parallel for
#endif
		for(int elementIndex = 0; elementIndex < elementCount; elementIndex++)
		{
			const T& e = elementPtr[elementIndex];
			UINT closestClusterIndex = 0;
			double closestClusterDist = Metric::Dist(e, clustersPtr[0].center);
			for(int clusterIndex = 1; clusterIndex < clusterCount; clusterIndex++)
			{
				double curClusterDist = Metric::Dist(e, clustersPtr[clusterIndex].center);
				if(curClusterDist < closestClusterDist)
				{
					closestClusterIndex = clusterIndex;
					closestClusterDist = curClusterDist;
				}
			}
			storage[elementIndex] = closestClusterIndex;
		}

		for(int elementIndex = 0; elementIndex < elementCount; elementIndex++)
			clustersPtr[storage[elementIndex]].addEntry(elementPtr[elementIndex]);

		for(int clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++)
			clustersPtr[clusterIndex].finalizeIteration(elements[rand()%elements.size()]);
	}

	void iterate(const std::vector<T> &elements, const std::vector<T> &weightedElements, const std::vector<float> &weights, std::vector<UINT> &storage)
	{
		const int elementCount = (int)elements.size();
		const T* elementPtr = &elements[0];
		const float* weightsPtr = &weights[0];
		const UINT clusterCount = (UINT)m_clusters.size();
		KMeansCluster<T> *clustersPtr = &m_clusters[0];

#ifdef MLIB_OPENMP
#pragma omp parallel for
#endif
		for(int elementIndex = 0; elementIndex < elementCount; elementIndex++)
		{
			const T& e = elementPtr[elementIndex];
			UINT closestClusterIndex = 0;
			double closestClusterDist = Metric::Dist(e, clustersPtr[0].center);
			for(UINT clusterIndex = 1; clusterIndex < clusterCount; clusterIndex++)
			{
				double curClusterDist = Metric::Dist(e, clustersPtr[clusterIndex].center);
				if(curClusterDist < closestClusterDist)
				{
					closestClusterIndex = clusterIndex;
					closestClusterDist = curClusterDist;
				}
			}
			storage[elementIndex] = closestClusterIndex;
		}

		for(int elementIndex = 0; elementIndex < elementCount; elementIndex++)
			clustersPtr[storage[elementIndex]].addEntry(weightedElements[elementIndex], weightsPtr[elementIndex]);

		for(UINT clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++)
			clustersPtr[clusterIndex].finalizeIteration(ml::util::randomElement(elements));
	}

	std::vector< KMeansCluster<T> > m_clusters;
};

}  // namespace ml

#endif  // CORE_MATH_KMEANSCLUSTERING_H_