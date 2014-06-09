#pragma once

/************************************************************************/
/* Timings logger                                                       */
/************************************************************************/

#define BENCHMARK_SAMPLES 128

class TimingLog
{	
	public:

		//! prints all the timings meassured so far
		static void printTimings();

		//! resets all the timings reset so far
		static void resetTimings();


		static double totalTimeBilateralFiltering;
		static unsigned int countBilateralFiltering;

		static double totalTimeTrackCamera; 
		static unsigned int countTrackCamera;

		static double totalTimeRenderScene;
		static unsigned int countRenderScene;

		static double totalTimeUpdateScene;
		static unsigned int countUpdateScene;

		static double totalTimeCompactifyAppendBuffer;
		static unsigned int countCompactifyAppendBuffer;

		static double totalTimeRemoveAndIntegrate;
		static unsigned int countRemoveAndIntegrate;

		static double totalTimeRayCast;
		static unsigned int countRayCast;

		static double totalTimeFillVoxelGrid;
		static unsigned int countFillVoxelGrid;

		static double totalTimeRayIntervalSplatting;
		static unsigned int countRayIntervalSplatting;

		static double totalTimeRayMarchingStepsSplatting;
		static unsigned int countRayMarchingStepsSplatting;

		static double totalTimeAlloc;
		static unsigned int countAlloc;

		static double totalTimeIntegrate;
		static unsigned int countIntegrate;

		static double totalTimeGarbageCollect0;
		static unsigned int countGarbageCollect0;

		static double totalTimeGarbageCollect1;
		static unsigned int countGarbageCollect1;

		static double totalTimeReductionCPU;
		static unsigned int countReductionCPU; 

		static double totalTimeStreamOut;
		static unsigned int countStreamOut;

		static double totalTimeStreamIn;
		static unsigned int countStreamIn;		

		static double totalTimeCompactifyHash;
		static unsigned int countCompactifyHash;

		static double totalTimeErode;
		static unsigned int countErode;



		////benchmark
		static double totalTimeAllAvgArray[BENCHMARK_SAMPLES];
		static unsigned int countTotalTimeAll;
		static double totalTimeAllWorst;
		static double totalTimeAllMaxAvg;
		static double totalTimeAllMinAvg;


		static double totalTimeMisc;
		static unsigned int countTimeMisc;

		static double totalTimeRender;
		static unsigned int countTimeRender;

		static double totalTimeTrack;
		static unsigned int countTimeTrack;

		static double totalTimeSceneUpdate;
		static unsigned int countTimeSceneUpdate;

		static double totalTimeAll;
		static double totalTimeSquaredAll;
};
