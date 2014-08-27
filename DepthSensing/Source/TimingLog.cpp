#include "stdafx.h"

#include "TimingLog.h"


#include "GlobalAppState.h"
#include <iostream>

double TimingLog::totalTimeBilateralFiltering = 0.0;
unsigned int TimingLog::countBilateralFiltering = 0;

double TimingLog::totalTimeTrackCamera = 0.0;
unsigned int TimingLog::countTrackCamera = 0;

double TimingLog::totalTimeRenderScene = 0.0;
unsigned int TimingLog::countRenderScene = 0;

double TimingLog::totalTimeUpdateScene = 0.0;
unsigned int TimingLog::countUpdateScene = 0;

double TimingLog::totalTimeCompactifyAppendBuffer = 0.0;
unsigned int TimingLog::countCompactifyAppendBuffer = 0;

double TimingLog::totalTimeRemoveAndIntegrate = 0.0;
unsigned int TimingLog::countRemoveAndIntegrate = 0;

double TimingLog::totalTimeRayCast = 0.0;
unsigned int TimingLog::countRayCast = 0;

double TimingLog::totalTimeFillVoxelGrid = 0.0;
unsigned int TimingLog::countFillVoxelGrid = 0;

double TimingLog::totalTimeRayIntervalSplatting = 0.0;
unsigned int TimingLog::countRayIntervalSplatting = 0;

double TimingLog::totalTimeRayMarchingStepsSplatting = 0.0;
unsigned int TimingLog::countRayMarchingStepsSplatting = 0;

double TimingLog::totalTimeAlloc = 0.0;
unsigned int TimingLog::countAlloc = 0;

double TimingLog::totalTimeIntegrate = 0.0;
unsigned int TimingLog::countIntegrate = 0;

double TimingLog::totalTimeGarbageCollect0 = 0.0;
unsigned int TimingLog::countGarbageCollect0 = 0;

double TimingLog::totalTimeGarbageCollect1 = 0.0;
unsigned int TimingLog::countGarbageCollect1 = 0;

double TimingLog::totalTimeReductionCPU = 0.0;
unsigned int TimingLog::countReductionCPU = 0;

double TimingLog::totalTimeStreamOut = 0.0;
unsigned int TimingLog::countStreamOut = 0;

double TimingLog::totalTimeStreamIn = 0.0;
unsigned int TimingLog::countStreamIn = 0;

double TimingLog::totalTimeCompactifyHash = 0.0;
unsigned int TimingLog::countCompactifyHash = 0;

double TimingLog::totalTimeErode = 0.0;
unsigned int TimingLog::countErode = 0;

double TimingLog::totalTimeAll = 0.0;
double TimingLog::totalTimeSquaredAll = 0.0;


/////////////
// benchmark
/////////////

double TimingLog::totalTimeAllAvgArray[BENCHMARK_SAMPLES];
unsigned int TimingLog::countTotalTimeAll = 0;
double TimingLog::totalTimeAllWorst = 0.0;
double TimingLog::totalTimeAllMaxAvg = 0.0;
double TimingLog::totalTimeAllMinAvg = 0.0;


double TimingLog::totalTimeMisc = 0.0;
unsigned int TimingLog::countTimeMisc = 0;

double TimingLog::totalTimeRender = 0.0;
unsigned int TimingLog::countTimeRender = 0;

double TimingLog::totalTimeTrack = 0.0;
unsigned int TimingLog::countTimeTrack = 0;

double TimingLog::totalTimeSceneUpdate = 0.0;
unsigned int TimingLog::countTimeSceneUpdate = 0;


void TimingLog::printTimings()
{
	if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		if (countBilateralFiltering != 0) std::cout << "Bilateral Filter: " <<  totalTimeBilateralFiltering/countBilateralFiltering << std::endl;
		if (countTrackCamera != 0) std::cout << "Track Camera: " <<  totalTimeTrackCamera/countTrackCamera << std::endl;
		if (countRenderScene != 0) std::cout << "Render Scene: " << totalTimeRenderScene/countRenderScene << std::endl;
		if (countUpdateScene != 0)	std::cout << "Update Scene: " << totalTimeUpdateScene/countUpdateScene << std::endl;
		if (countCompactifyAppendBuffer != 0) std::cout << "Compactify with Append Buffer: " << totalTimeCompactifyAppendBuffer/countCompactifyAppendBuffer << std::endl;
		if (countRemoveAndIntegrate != 0 ) std::cout << "Remove and Integrate: " << totalTimeRemoveAndIntegrate/countRemoveAndIntegrate << std::endl;
		if (countRayCast != 0 ) std::cout << "RayCast: " << totalTimeRayCast/countRayCast << std::endl;
		if (countRayIntervalSplatting != 0 ) std::cout << "RayIntervalSplatting: " << totalTimeRayIntervalSplatting/countRayIntervalSplatting << std::endl;
		if (countRayMarchingStepsSplatting != 0 ) std::cout << "RayMarchingSplatting: " << totalTimeRayMarchingStepsSplatting/countRayMarchingStepsSplatting << std::endl;
		if (countFillVoxelGrid != 0 ) std::cout << "FillVoxelGrid: " << totalTimeFillVoxelGrid/countFillVoxelGrid << std::endl;
		if (countAlloc != 0) std::cout << "Alloc: " << totalTimeAlloc/countAlloc << std::endl;
		if (countIntegrate != 0 ) std::cout << "Integrate: " << totalTimeIntegrate/countIntegrate << std::endl;
		if (countReductionCPU != 0 ) std::cout << "ReductionCPU: " << totalTimeReductionCPU/countReductionCPU << std::endl;
		if (countGarbageCollect0 != 0) std::cout << "GarbageCollect0: " << totalTimeGarbageCollect0/countGarbageCollect0 << std::endl;
		if (countGarbageCollect1 != 0) std::cout << "GarbageCollect1: " << totalTimeGarbageCollect1/countGarbageCollect1 << std::endl;
		if (countErode != 0) std::cout << "Erode: " << totalTimeErode/countErode << std::endl;

		if (countStreamOut != 0)	std::cout << "StreamOut: " << totalTimeStreamOut/countStreamOut << std::endl;
		if (countStreamIn != 0)		std::cout << "StreamIn: " << totalTimeStreamIn/countStreamIn << std::endl;

		if (countCompactifyHash != 0) std::cout << "Compactify Hash: " << totalTimeCompactifyHash/countCompactifyHash << std::endl;

		std::cout << std::endl; std::cout << std::endl;
	}

	if (GlobalAppState::getInstance().s_timingsTotalEnabled) {
		if (countTotalTimeAll != 0) {
			double avg = 0.0;
			for (UINT i = 0; i < std::min((unsigned int)BENCHMARK_SAMPLES, countTotalTimeAll); i++) {
				avg += totalTimeAllAvgArray[i];
			}
			avg /= std::min((unsigned int)BENCHMARK_SAMPLES, countTotalTimeAll);
			if (countTotalTimeAll >= BENCHMARK_SAMPLES) {
				totalTimeAllMaxAvg = std::max(totalTimeAllMaxAvg, avg);
				totalTimeAllMinAvg = std::min(totalTimeAllMinAvg, avg);
			}
			std::cout << "count: " << countTotalTimeAll << std::endl;
			std::cout << "Time All Avg:\t" << avg << std::endl;
			std::cout << "Time All Avg Total:\t" << totalTimeAll/countTotalTimeAll << std::endl;
			double deviation = sqrt((totalTimeSquaredAll/countTotalTimeAll)-(totalTimeAll/countTotalTimeAll)*(totalTimeAll/countTotalTimeAll));
			std::cout << "Time All Std.Dev.:\t" << deviation << std::endl;

			std::cout << "Time All Worst:\t" << totalTimeAllWorst << std::endl;
			//std::cout << "Time All MaxAvg:\t" << totalTimeAllMaxAvg << std::endl;
			//std::cout << "Time All MinAvg:\t" << totalTimeAllMinAvg << std::endl;
			std::cout << std::endl;
		}
	}
	if (GlobalAppState::getInstance().s_timingsStepsEnabled) {
		double total = totalTimeMisc + totalTimeRender + totalTimeSceneUpdate + totalTimeTrack;
		if (countTimeMisc != 0) std::cout << "Time Misc:\t" << totalTimeMisc/countTimeMisc << "\t % " << 100.0*totalTimeMisc / total << std::endl;
		if (countTimeRender != 0) std::cout << "Time Render:\t" << totalTimeRender/countTimeRender << "\t % " << 100.0*totalTimeRender / total  << std::endl;
		if (countTimeSceneUpdate != 0) std::cout << "Time Update:\t" << totalTimeSceneUpdate/countTimeSceneUpdate << "\t % " << 100.0*totalTimeSceneUpdate / total << std::endl;
		if (countTimeTrack != 0) std::cout << "Time Track:\t" << totalTimeTrack/countTimeTrack << "\t % " << 100.0*totalTimeTrack / total << std::endl;
		std::cout << std::endl;
	}
}

void TimingLog::resetTimings()
{
	totalTimeBilateralFiltering = 0.0;
	countBilateralFiltering = 0;

	totalTimeTrackCamera = 0.0; 
	countTrackCamera = 0;

	totalTimeRenderScene = 0.0;
	countRenderScene = 0;

	totalTimeUpdateScene = 0.0;
	countUpdateScene = 0;

	totalTimeCompactifyAppendBuffer = 0.0;
	countCompactifyAppendBuffer = 0;

	totalTimeRemoveAndIntegrate = 0.0;
	countRemoveAndIntegrate = 0;

	totalTimeRayCast= 0.0;
	countRayCast = 0;

	totalTimeFillVoxelGrid= 0.0;
	countFillVoxelGrid = 0;

	totalTimeRayIntervalSplatting = 0.0;
	countRayIntervalSplatting = 0;

	totalTimeRayMarchingStepsSplatting = 0.0;
	countRayMarchingStepsSplatting = 0;

	totalTimeAlloc = 0.0;
	countAlloc = 0;

	totalTimeIntegrate = 0.0;
	countIntegrate = 0;

	totalTimeGarbageCollect0 = 0.0;
	countGarbageCollect0 = 0;

	totalTimeGarbageCollect1 = 0.0;
	countGarbageCollect1 = 0;

	totalTimeReductionCPU = 0.0;
	countReductionCPU = 0; 

	totalTimeStreamOut = 0.0;
	countStreamOut = 0;

	totalTimeStreamIn = 0.0;
	countStreamIn = 0;

	totalTimeCompactifyHash = 0.0;
	countCompactifyHash = 0;

	totalTimeErode = 0.0;
	countErode = 0;



	for (UINT i = 0; i < BENCHMARK_SAMPLES; i++) totalTimeAllAvgArray[i] = 0.0;
	countTotalTimeAll = 0;
	totalTimeAllWorst = 0.0;
	totalTimeAllMaxAvg = 0.0;
	totalTimeAllMinAvg = std::numeric_limits<double>::max();

	totalTimeMisc = 0.0;
	countTimeMisc = 0;

	totalTimeRender = 0.0;
	countTimeRender = 0;

	totalTimeTrack = 0.0;
	countTimeTrack = 0;

	totalTimeSceneUpdate = 0.0;
	countTimeSceneUpdate = 0;

	totalTimeAll = 0.0;
	totalTimeSquaredAll = 0.0;
}
