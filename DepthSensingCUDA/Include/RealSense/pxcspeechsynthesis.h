/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2013-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
/**
    @file pxcspeechsynthesis.h
 */
#pragma once
#include "pxcaudio.h"

/**
   This class defines a standard interface for performing speech synthesis.
 */
class PXCSpeechSynthesis: public PXCBase {
public:

    PXC_CUID_OVERWRITE(PXC_UID('V','I','T','S'));

    /**
        @enum LanguageType
        Enumerate all supported languages.
    */
    enum LanguageType {
        LANGUAGE_US_ENGLISH     = PXC_UID('e','n','U','S'),        /** US English */
        LANGUAGE_GB_ENGLISH     = PXC_UID('e','n','G','B'),        /** British English */
        LANGUAGE_DE_GERMAN      = PXC_UID('d','e','D','E'),        /** German */
        LANGUAGE_US_SPANISH     = PXC_UID('e','s','U','S'),        /** Spanish */
        LANGUAGE_FR_FRENCH      = PXC_UID('f','r','F','R'),        /** French */
        LANGUAGE_IT_ITALIAN     = PXC_UID('i','t','I','T'),        /** Italian */
        LANGUAGE_JP_JAPANESE    = PXC_UID('j','a','J','P'),        /** Japanese */
        LANGUAGE_CN_CHINESE     = PXC_UID('z','h','C','N'),        /** Simplified Chinese */
        LANGUAGE_BR_PORTUGUESE  = PXC_UID('p','t','B','R'),        /** Portuguese */
    };

    /** 
        @enum VoiceType
        Enumerate all supported voices.
    */
    enum VoiceType {
        VOICE_ANY=0,        /* Any available voice */
    };

    /**
        @struct ProfileInfo
        Describe the algorithm configuration parameters.
    */
    struct ProfileInfo {
        PXCAudio::AudioInfo outputs;        /** The synthesized audio format. Adjust bufferSize for the required latency. */
        LanguageType    language;           /** The supported language */
        VoiceType       voice;              /** The voice */
        pxcF32          rate;               /** The speaking speed. The default is 100. Smaller is slower and bigger is faster. */
        pxcI32          volume;             /** The speaking volume from 0 to 100 (loudest). */
		pxcI32          pitch;				/** default pitch is 100. range [50 to 200] */
        pxcI32          eosPauseDuration;   /** End of sentence wait duration. range [0 to 9 multiplied by 200msec] */
        pxcI32          reserved[4];
    };

    /**
        @brief The function returns the available algorithm configuration parameters.
        @param[in]  pidx        The zero-based index to retrieve all configuration parameters.
        @param[out] pinfo       The configuration parameters, to be returned.
        @return PXC_STATUS_NO_ERROR                Successful execution.
        @return PXC_STATUS_ITEM_UNAVAILABLE        No more configurations.
    */
    virtual pxcStatus PXCAPI QueryProfile(pxcI32 pidx, ProfileInfo *pinfo)=0;

    /**
        @brief The function returns the current working algorithm configuration parameters.
        @param[out] pinfo       The configuration parameters, to be returned.
        @return PXC_STATUS_NO_ERROR                Successful execution.
    */
    pxcStatus __inline QueryProfile(ProfileInfo *pinfo) { return QueryProfile(WORKING_PROFILE,pinfo); }

    /**
        @brief The function sets the current working algorithm configuration parameters.
        @param[in] pinfo        The configuration parameters.
        @return PXC_STATUS_NO_ERROR                Successful execution.
    */
    virtual pxcStatus PXCAPI SetProfile(ProfileInfo *pinfo)=0;

    /**
        @brief The function synthesizes the sentence for later use. The function may take some time
        to generate the fully synthesized speech.
        @param[in] sid          The sentence identifier. Can be any non-zero unique number.
        @param[in] sentence     The sentence string.
        @return PXC_STATUS_NO_ERROR        Successful execution.
    */
    virtual pxcStatus PXCAPI BuildSentence(pxcUID sid, pxcCHAR *sentence)=0;

    /**
        @brief The function retrieves the PXCAudio buffer for the specified sentence. There could be more
        than one PXCAudio buffer. The application should keep retrieving with increased index, until the 
        function returns NULL. The audio buffer is internally managed. Do not release the instance.
        @param[in] sid          The sentence identifier.
        @param[in] idx          The zero-based index to retrieve multiple samples.
        @return the Audio buffer, or NULL if there is no more.
    */
    virtual PXCAudio* PXCAPI QueryBuffer(pxcUID sid, pxcI32 idx)=0;

    /**
        @brief The function returns the number of PXCAudio buffers used for the specified 
        synthesized sentence.
        @param[in] sid          The sentence identifier.
        @return the number of PXCAudio buffers, or 0 if the sentence is not found.
    */
    virtual pxcI32 PXCAPI QueryBufferNum(pxcUID sid)=0;

    /**
        @brief The function returns the number of audio samples for the specified synthesized sentence. 
        Each audio sample consists of multiple channels according to the format definition.
        @param[in] sid          The sentence identifier.
        @return the sample number, or 0 if the sentence is not found.
    */
    virtual pxcI32 PXCAPI QuerySampleNum(pxcUID sid)=0;

    /**
        @brief The function releases any resources allocated for the sentence identifier.
        @param[in] sid          The sentence identifier.
    */
    virtual void PXCAPI ReleaseSentence(pxcUID sid)=0;
};
