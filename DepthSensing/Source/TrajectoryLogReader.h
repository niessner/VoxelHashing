#pragma once

/************************************************************************/
/* Can read pre-recorded trajectories and use it for surface fusion     */
/************************************************************************/

#include "stdafx.h"

#include <string>
#include <fstream>
#include <iostream>
#include <list>

class TrajectoryLogReader
{
public:
	TrajectoryLogReader(void) {
		m_CurrFrame = 0;
	}
	~TrajectoryLogReader(void) {

	}

	void Init(const std::string& filename) {
		std::ifstream in;
		in.open(filename);
		m_Transforms.clear();
		//m_Transforms.push_back(mat4f(mat4f::Identity));
		while (!in.eof()) {
			int currMatrix;
			int dummy0, dummy1;
			in >> currMatrix >> dummy0 >> dummy1;
			float data[16]; 
			for (unsigned int i = 0; i < 16; i++) {
				in >> data[i];
			}
			mat4f t(data);
			//weird coordinate system switch - but it seems we need that...
			t(0,1) *= -1;
			t(1,0) *= -1;
			t(1,2) *= -1;
			t(1,3) *= -1;
			t(2,1) *= -1;
			m_Transforms.push_back(t);
			//std::cout << m_Transforms.front() << std::endl;
		}


		in.close();
	}


	mat4f getNextTransform() {
		m_CurrFrame++;
		if (m_Transforms.empty()) {
			return mat4f::identity();
		}
		
		m_Transforms.push_back(m_Transforms.front());
		m_Transforms.pop_front();
		return m_Transforms.back();
	}

private:
	unsigned int	m_CurrFrame;
	std::list<mat4f> m_Transforms;
};


