
#include "stdafx.h"

#include "TimingLog.h"

double TimingLog::totalTimeHoleFilling = 0.0;
unsigned int TimingLog::countTimeHoleFilling = 0;

double TimingLog::totalTimeRender = 0.0;
unsigned int TimingLog::countTimeRender = 0;

double TimingLog::totalTimeOptimizer = 0.0;
unsigned int TimingLog::countTimeOptimizer = 0;

double TimingLog::totalTimeFilterColor = 0.0;
unsigned int TimingLog::countTimeFilterColor = 0;

double TimingLog::totalTimeFilterDepth = 0.0;
unsigned int TimingLog::countTimeFilterDepth = 0;

double TimingLog::totalTimeRGBDAdapter = 0.0;
unsigned int TimingLog::countTimeRGBDAdapter = 0;

double TimingLog::totalTimeClusterColor = 0.0;
unsigned int TimingLog::countTimeClusterColor = 0;

double TimingLog::totalTimeEstimateLighting = 0.0;
unsigned int TimingLog::countTimeEstimateLighting = 0;

double TimingLog::totalTimeRemapDepth = 0.0;
unsigned int TimingLog::countTimeRemapDepth = 0;

double TimingLog::totalTimeSegment = 0.0;
unsigned int TimingLog::countTimeSegment = 0;

double TimingLog::totalTimeTracking = 0.0;
unsigned int TimingLog::countTimeTracking = 0;

double TimingLog::totalTimeSFS = 0.0;
unsigned int TimingLog::countTimeSFS = 0;

double TimingLog::totalTimeRayCast = 0.0;
unsigned int TimingLog::countTimeRayCast = 0;

double TimingLog::totalTimeRayIntervalSplatting = 0.0;
unsigned int TimingLog::countTimeRayIntervalSplatting = 0;

//double TimingLog::totalTimeRayIntervalSplattingCUDA = 0.0;
//unsigned int TimingLog::countTimeRayIntervalSplattingCUDA = 0;
//
//double TimingLog::totalTimeRayIntervalSplattingDX11 = 0.0;
//unsigned int TimingLog::countTimeRayIntervalSplattingDX11 = 0;

double TimingLog::totalTimeCompactifyHash = 0.0;
unsigned int TimingLog::countTimeCompactifyHash = 0;

double TimingLog::totalTimeAlloc = 0.0;
unsigned int TimingLog::countTimeAlloc = 0;

double TimingLog::totalTimeIntegrate = 0.0;
unsigned int TimingLog::countTimeIntegrate = 0;

/////////////
// benchmark
/////////////

double TimingLog::totalTimeAllAvgArray[BENCHMARK_SAMPLES];
unsigned int TimingLog::countTotalTimeAll = 0;
double TimingLog::totalTimeAllWorst = 0.0;
double TimingLog::totalTimeAllMaxAvg = 0.0;
double TimingLog::totalTimeAllMinAvg = 0.0;
double TimingLog::totalTimeAll = 0.0;
double TimingLog::totalTimeSquaredAll = 0.0;
