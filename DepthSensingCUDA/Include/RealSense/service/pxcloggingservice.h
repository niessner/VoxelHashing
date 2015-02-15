/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#pragma once
#include "pxcbase.h"
#include <string>

class PXCLoggingService: public PXCBase {
public:
    PXC_CUID_OVERWRITE(0x3870DB28);

    typedef pxcI32 Level;
    enum {
        LEVEL_FATAL = 50000,
        LEVEL_ERROR = 40000,
        LEVEL_WARN  = 30000,
        LEVEL_INFO  = 20000,
        LEVEL_DEBUG = 10000,
        LEVEL_TRACE =  5000,
    };

    enum ConfigMode {
        CONFIG_DEFAULT              = 0x1,
        CONFIG_PROPERTY_FILE_LOG4J  = 0x2,
        CONFIG_XML_FILE_LOG4J       = 0x4,
    };

    virtual pxcStatus   PXCAPI SetLoggerName(const pxcCHAR* name)=0; // name in loggers hierarchy. NULL means root logger. Name may contain dots like class and namespace hierarchy in C#
    virtual pxcStatus   PXCAPI Configure(ConfigMode configMode, const pxcCHAR* config, int fileWatchDelay)=0; // if fileWatchDelay non-zero, specifies delay in ms to check if config file changed (only for CONFIG_PROPERTY_FILE_LOG4J and CONFIG_XML_FILE_LOG4J)
    virtual pxcBool     PXCAPI IsConfigured()=0; // configuration is process-wide for all loggers, call Configure() once per application
    virtual pxcStatus   PXCAPI SetLevel(Level level)=0; // overwrite level specified in initial configuration

    virtual pxcBool     PXCAPI IsLevelEnabled(Level level)=0;
    virtual Level       PXCAPI GetLevel()=0;
    virtual void        PXCAPI Log (Level level, const char*    message, const char* fileName, int lineNumber, const char* functionName)=0;
    virtual void        PXCAPI LogW(Level level, const pxcCHAR* message, const char* fileName, int lineNumber, const char* functionName)=0;

    // performance tracing
    virtual void        PXCAPI TaskBegin(Level level, const char* taskName)=0;
    virtual void        PXCAPI TaskEnd(Level level, const char* taskName)=0;
};

////////////////////////////////////////////////////////////////////////////////

// Utility macro for logging
#define PXCLOG(_logger, _level, _message)               \
{                                                       \
    if (_logger->IsLevelEnabled(_level))                \
    {                                                   \
        _logger->Log(_level, _message, __FILE__, __LINE__, __FUNCSIG__); \
    }                                                   \
}

// Utility class for tasks in performance tracing
class PXCTraceTask
{
public:
    PXCTraceTask(PXCLoggingService *logger, PXCLoggingService::Level level, const char* taskName)
    {
        if (logger->IsLevelEnabled(level) && taskName[0])
        {
            this->logger = logger;
            this->level = level;
            this->taskName = taskName;
            logger->TaskBegin(level, taskName);
        } else
        {
            this->logger = NULL;
        }
    }
    ~PXCTraceTask()
    {
        if (this->logger)
        {
            logger->TaskEnd(level, taskName);
        }
    }
private:
    PXCLoggingService*         logger;
    PXCLoggingService::Level   level;
    const char*     taskName;
};
