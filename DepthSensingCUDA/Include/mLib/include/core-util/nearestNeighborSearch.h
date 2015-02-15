
#ifndef CORE_UTIL_NEARESTNEIGHBORSEARCH_H_
#define CORE_UTIL_NEARESTNEIGHBORSEARCH_H_

namespace ml
{

template<class FloatType>
class NearestNeighborSearch
{
public:
	void init(const std::vector< const FloatType* > &points, UINT dimension, UINT maxK)
	{
		initInternal(points, dimension, maxK);
	}
	void kNearest(const FloatType *query, UINT k, FloatType epsilon, std::vector<UINT> &result) const
	{
		kNearestInternal(query, k, epsilon, result);
	}
    UINT nearest(const FloatType *query) const
    {
        std::vector<UINT> result;
        kNearestInternal(query, 1, 0.0f, result);
        return result[0];
    }
	
	void init(const std::vector< std::vector<FloatType> > &points, UINT maxK)
	{
		//Vector< const D* > v = points.map([](const Vector<D> &x) {return x.ptr();});
		std::vector< const FloatType* > v(points.size());
		for(UINT i = 0; i < points.size(); i++)
			v[i] = &points[i][0];
		init(v, (UINT)points[0].size(), maxK);
	}

	void kNearest(const std::vector<FloatType> &query, UINT k, FloatType epsilon, std::vector<UINT> &result) const
	{
		kNearestInternal(&query[0], k, epsilon, result);
	}

	std::vector<UINT> kNearest(const std::vector<FloatType> &query, UINT k, FloatType epsilon) const
	{
		std::vector<UINT> result;
		kNearestInternal(&query[0], k, epsilon, result);
		return result;
	}

	std::vector<UINT> kNearest(const FloatType *query, UINT k, FloatType epsilon) const
	{
		std::vector<UINT> result;
		kNearestInternal(query, k, epsilon, result);
		return result;
	}

    std::vector<UINT> fixedRadius(const std::vector<FloatType> &query, UINT k, FloatType radiusSq) const
	{
		std::vector<UINT> result;
		fixedRadiusInternal(&query[0], k, radiusSq, 0.0f, result);
		return result;
	}

    std::vector<UINT> fixedRadius(const FloatType *query, UINT k, FloatType radiusSq) const
    {
        std::vector<UINT> result;
        fixedRadiusInternal(query, k, radiusSq, 0.0f, result);
        return result;
    }

    std::vector< std::pair<UINT, FloatType> > fixedRadiusDist(const FloatType *query, UINT k, FloatType radiusSq) const
    {
        std::vector< std::pair<UINT, FloatType> > result;
        fixedRadiusInternalDist(query, k, radiusSq, 0.0f, result);
        return result;
    }

private:
	virtual void initInternal(const std::vector< const FloatType* > &points, UINT dimension, UINT maxK) = 0;
	virtual void kNearestInternal(const FloatType *query, UINT k, FloatType epsilon, std::vector<UINT> &result) const = 0;
    virtual void fixedRadiusInternal(const FloatType *query, UINT k, FloatType radiusSq, FloatType epsilon, std::vector<UINT> &result) const = 0;
    virtual void fixedRadiusInternalDist(const FloatType *query, UINT k, FloatType radiusSq, FloatType epsilon, std::vector< std::pair<UINT, FloatType> > &result) const = 0;
};

template<class FloatType>
class KNearestNeighborQueue
{
public:
	KNearestNeighborQueue() {}

	struct NeighborEntry
	{
		NeighborEntry() {}
		NeighborEntry(int _index, FloatType _dist)
		{
			index = _index;
			dist = _dist;
		}
		int index;
		FloatType dist;
	};

    KNearestNeighborQueue(UINT k, FloatType clearValue)
    {
        init(k, clearValue);
    }

	void init(UINT k, FloatType clearValue)
	{
		if(m_queue.size() != k) m_queue.resize(k);
		clear(clearValue);
	}

	void clear(FloatType clearValue)
	{
		m_queue.assign(m_queue.size(), NeighborEntry(-1, clearValue));
		m_farthestDist = clearValue;
	}

    inline void insert(int index, FloatType dist)
    {
        insert(NeighborEntry(index, dist));
    }

	inline void insert(const NeighborEntry &entry)
	{
		if(entry.dist < m_farthestDist)
		{
			m_queue.back() = entry;
			const int queueLength = (int)m_queue.size();
			if(queueLength > 1)
			{
				NeighborEntry *data = &m_queue[0];
				for(int index = queueLength - 2; index >= 0; index--)
				{
					if(data[index].dist > data[index + 1].dist)
					{
						std::swap(data[index], data[index + 1]);
					}
				}
			}
			m_farthestDist = m_queue.back().dist;
		}
	}

	const std::vector<NeighborEntry>& queue() const
	{
		return m_queue;
	}

private:
	FloatType m_farthestDist;
	std::vector<NeighborEntry> m_queue;
};

template<class FloatType>
class NearestNeighborSearchBruteForce : public NearestNeighborSearch<FloatType>
{
public:
	NearestNeighborSearchBruteForce() {}

	void initInternal(const std::vector< const FloatType* > &points, UINT dimension, UINT maxK)
	{
		m_dimension = dimension;
		m_pointData.resize(points.size() * dimension);
		int pointIndex = 0;
		for(auto p : points)
		{
			m_points.push_back(&m_pointData[0] + pointIndex);
			for(UINT d = 0; d < m_dimension; d++)
				m_pointData[pointIndex++] = p[d];
		}
	}

	void kNearestInternal(const FloatType *query, UINT k, FloatType epsilon, std::vector<UINT> &result) const
	{
		m_queue.init(k, std::numeric_limits<FloatType>::max());
		
		for(UINT pointIndex = 0; pointIndex < m_points.size(); pointIndex++)
		{
			FloatType dist = 0.0f;
			const FloatType* point = m_points[pointIndex];
			for(UINT d = 0; d < m_dimension; d++)
			{
				FloatType diff = point[d] - query[d];
				dist += diff * diff;
			}
			m_queue.insert(typename KNearestNeighborQueue<FloatType>::NeighborEntry(pointIndex, dist));
		}

		if(result.size() != k) result.resize(k);
		UINT resultIndex = 0;
		for(const auto &e : m_queue.queue())
		{
			result[resultIndex++] = e.index;
		}
	}

    void fixedRadiusInternal(const FloatType *query, UINT k, FloatType radiusSq, FloatType epsilon, std::vector<UINT> &result) const
    {
        throw MLIB_EXCEPTION("fixedRadiusInternal not implemented");
    }

    void fixedRadiusInternalDist(const FloatType *query, UINT k, FloatType radiusSq, FloatType epsilon, std::vector< std::pair<UINT, FloatType> > &result) const
    {
        throw MLIB_EXCEPTION("fixedRadiusInternalDist not implemented");
    }

private:
	UINT m_dimension;
	std::vector<FloatType> m_pointData;
	std::vector< const FloatType* > m_points;
	mutable KNearestNeighborQueue<FloatType> m_queue;
};

}  // namespace ml

#endif  // CORE_UTIL_NEARESTNEIGHBORSEARCH_H_
