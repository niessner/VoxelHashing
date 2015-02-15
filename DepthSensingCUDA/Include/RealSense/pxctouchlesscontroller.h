/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2013 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#pragma once
#include "pxccapture.h"
#include "pxchandmodule.h"
#include "pxchanddata.h"
/**
 @class	PXCTouchlessController

 @brief	This module interpert user actions to UX commands intended to control a windows 8 computer, 
  such as scroll, zoom, select atile etc. The module fire events for each such action as well as inject 
  touch, mouse and keyboard event to the operating system to perform the action. 
  Developer may listen to such events to enable application specific reactions, or just enable this module 
  and depend on the normal OS reactions to the injected events.
  There is also a WPF dll that provides default visual feedback which may be easily linked to those events, 
  see touchless_controller_visual_feedback sample.
  Few configuration options are available for developers to influence the way the module opareate, 
  like enabling or disabling specific behaviors.
 */

class PXCTouchlessController: public PXCBase {
public:

    PXC_CUID_OVERWRITE(PXC_UID('F','L','K','S'));

    struct ProfileInfo
    {
		typedef pxcEnum Configuration; // an or value of UX options relevant to specific application
        enum {
			Configuration_None = 0x00000000, // No option is selected - use default behavior 
			Configuration_Allow_Zoom = 0x00000001, // Should zoom be allowed
			Configuration_Use_Draw_Mode = 0x00000002, // Use draw mode - should be used for applications the need continues interaction (touch + movement) like drawing
			Configuration_Scroll_Horizontally = 0x00000004, // Enable horizontal scrolling using pinch gesture
			Configuration_Scroll_Vertically = 0x00000008, // Enable vertical scrolling  using pinch gesture
			Configuration_Meta_Context_Menu = 0x00000010, // Should Meta menu events be fired, triggered by v gesture
			Configuration_Enable_Injection = 0x00000020, // Disable the injection of keyboar/mouse/touch events
			Configuration_Edge_Scroll_Horizontally = 0x00000040, // Enable horizontal scrolling when pointer is on the edge of the screen
			Configuration_Edge_Scroll_Vertically = 0x00000080, // Enable vertical scrolling  when pointer is on the edge of the screen
			Configuration_Hide_Cursor_After_Touch_Injection = 0x00000100, // Should windows cursor be hidden after touch injection - other wise windows will make the cursor reappear
			Configuration_Allow_Back = 0x00000200 //  Enable Back Gesture
		};

		
        PXCHandModule*		handModule;   //the HandAnalysis module used by this module, dont set it when using SenseManager - this is just an output parameter
        Configuration       config;   // An or value of configuration options
    };
   
    /** 
        @brief Return the configuration parameters of the SDK's TouchlessController
        @param[out] pinfo the profile info structure of the configuration parameters.
        @return PXC_STATUS_NO_ERROR if the parameters were returned successfully; otherwise, return one of the following errors:
        PXC_STATUS_ITEM_UNAVAILABLE - Item not found/not available.\n
        PXC_STATUS_DATA_NOT_INITIALIZED - Data failed to initialize.\n                        
    */
    virtual pxcStatus  PXCAPI QueryProfile(ProfileInfo *pinfo)=0;

    /** 
        @brief Set configuration parameters of the SDK TouchlessController. 
        @param[in] pinfo the profile info structure of the configuration parameters.
        @return PXC_STATUS_NO_ERROR if the parameters were set correctly; otherwise, return one of the following errors:
        PXC_STATUS_INIT_FAILED - Module failure during initialization.\n
        PXC_STATUS_DATA_NOT_INITIALIZED - Data failed to initialize.\n                        
    */
    virtual pxcStatus  PXCAPI SetProfile(ProfileInfo *pinfo)=0;

	/*
		@breif Describe a UXEvent,
	*/
	struct UXEventData
	{
		/**
		 @enum	UXEventType
		
		 @brief	Values that represent UXEventType.
		 */
		enum UXEventType 
		{
			UXEvent_StartZoom,			// the user start performing a zoom operation - pan my also be performed during zoom
			UXEvent_Zoom,				// Fired while zoom operation is ongoing
			UXEvent_EndZoom,			// User stoped zoomig
			UXEvent_StartScroll,		// the user start performing a scroll or pan operation
			UXEvent_Scroll,				// Fired while scroll operation is ongoing
			UXEvent_EndScroll,			// User stoped scrolling (panning)
			UXEvent_StartDraw,			// User started drawing
			UXEvent_Draw,				// Fired while draw operation is ongoing
			UXEvent_EndDraw,			// User finshed drawing
			UXEvent_CursorMove,			// Cursor moved while not in any other mode
			UXEvent_Select,				// oser selected a button
			UXEvent_GotoStart,			// Got to windows 8 start screen
			UXEvent_CursorVisible,		// Cursor turned visible
			UXEvent_CursorNotVisible,	// Cursor turned invisible
			UXEvent_ReadyForAction,		// The user is ready to perform a zoom or scroll operation
			UXEvent_StartMetaCounter,   // Start Meta Menu counter visual
			UXEvent_StopMetaCounter,    // Abort Meta Menu Counter Visual
			UXEvent_ShowMetaMenu,       // Show Meta Menu
			UXEvent_HideMetaMenu,       // Hide Meta Menu
			UXEvent_MetaPinch,			// When a pinch was detected while in meta mode
			UXEvent_MetaOpenHand,       // When a pinch ends while in meta mode
			UXEvent_Back				// User perform back gesture
		};
		UXEventType type; // type of the event
		PXCPoint3DF32 position; // position where event happen values are in rang [0,1]
		PXCHandData::BodySideType bodySide; // the hand that issued the event

	};
	
	/**	
		@class UXEventHandler
		Interface for a callback for all categories of events 
	*/
	class UXEventHandler{
	public:
		/**
		@brief virtual destructor
		*/
		virtual ~UXEventHandler(){}
		/**
		 @brief The OnFiredUXEvent method is called when a UXWvent is fired.
		 @param[in] uxEventData contains all the information for the fired event.
		 @see UXEventData
		*/
		virtual  void PXCAPI OnFiredUXEvent(const UXEventData *uxEventData)=0;
	};


	/** 
    @brief Register an event handler object for UX Event. The event handler's OnFiredUXEvent method will be called each time a UX event is identified.
    @param[in] uxEventHandler a pointer to the event handler.
    @see UXEventHandler::OnFiredUXEvent
    @return PXC_STATUS_NO_ERROR if the registering an event handler was successful; otherwise, return the following error:
    PXC_STATUS_DATA_NOT_INITIALIZED - Data failed to initialize.\n        
    */
    virtual pxcStatus PXCAPI SubscribeEvent(UXEventHandler *uxEventHandler) = 0;

    /** 
        @brief Unsubscribe an event handler object for UX events.
        @param[in] uxEventHandler a pointer to the event handler that should be removed.
        @return PXC_STATUS_NO_ERROR if the unregistering the event handler was successful, an error otherwise.
    */
    virtual pxcStatus PXCAPI UnsubscribeEvent(UXEventHandler *uxEventHandler) = 0;

	/**
		 @struct	AlertData
	
		 @brief	An alert data, contain data describing an alert.
	*/
	struct AlertData
	{
		/**
		 @enum	AlertType
		
		 @brief	Values that represent AlertType.
		 */
		enum AlertType { 
			Alert_TooClose,         // The user hand is too close to the 3D camera
			Alert_TooFar,           // The user hand is too far from the 3D camera
			Alert_NoAlerts          // A previous alerted situation was ended
		};
		AlertType type; // the  type of the alert
	};

	/**	
		@class AlertHandler
		Interface for a callback for all categories of alerts 
	*/
	class AlertHandler{
	public:
		/**
		@brief virtual destructor
		*/
		virtual ~AlertHandler(){}
		/**
		 @brief The OnFiredAlert method is called when a registered alert event is fired.
		 @param[in] alertData contains all the information for the fired alert.
		 @see AlertData
		*/
		virtual  void PXCAPI OnFiredAlert(const AlertData *alertData)=0;
	};

	/** 
        @brief Register an event handler object for alerts. The event handler's OnFiredAlert method will be called each time an alert is identified.
        @param[in] alertHandler a pointer to the event handler.
        @see AlertHandler::OnFiredAlert
        @return PXC_STATUS_NO_ERROR if the registering an event handler was successful; otherwise, return the following error:
        PXC_STATUS_DATA_NOT_INITIALIZED - Data failed to initialize.\n        
    */
    virtual pxcStatus PXCAPI SubscribeAlert(AlertHandler *alertHandler) = 0;

    /** 
        @brief Unsubscribe an event handler object for alerts.
        @param[in] alertHandler a pointer to the event handler that should be removed.
        @return PXC_STATUS_NO_ERROR if the unregistering the event handler was successful, an error otherwise.
    */
    virtual pxcStatus PXCAPI UnsubscribeAlert(AlertHandler *alertHandler) = 0;

	/**
	 @enum	Action
	
	 @brief	Values that represent Action. Those are actions the module will inject to the OS
	 */
	enum Action 
		{
			Action_None=0,		// No action will be injected
			Action_LeftKeyPress,	// can be used to Go to the next item (Page/Slide/Photo etc.)
			Action_RightKeyPress,	//  can be used to Go to the previouse item (Page/Slide/Photo etc.)
			Action_BackKeyPress,	//  can be used to Go to the previouse item (Page/Slide/Photo etc.)
			Action_PgUpKeyPress,	//  can be used to Go to the previouse item (Page/Slide/Photo etc.)
			Action_PgDnKeyPress,	//  can be used to Go to the previouse item (Page/Slide/Photo etc.)
			Action_VolumeUp,
			Action_VolumeDown,
			Action_Mute,
			Action_NextTrack,
			Action_PrevTrack,
			Action_PlayPause,
			Action_Stop,
			Action_ToggleTabs,		// can be used to display tabs menu in Metro Internet Explorer
		};

	/**	
		@class AlertHandler
		Interface for a callback for all categories of actions 
	*/
	class ActionHandler {
	public:
		/**
		@brief virtual destructor
		*/
		virtual ~ActionHandler(){}
		/**
		 @brief The OnFiredAction method is called when a registered action mapping is triggered
		 @param[in] action the action that was fired
		 @see Action
		*/
		virtual  void PXCAPI OnFiredAction(const Action action)=0;
	};

	/**
	 @brief	Adds a gesture action mapping.
	 @param [in]	gestureName  	If non-null, name of the gesture.
	 @param	action					 	The action.
	 @param [in]	actionHandler	(Optional) If non-null, an action handler that will be called when the gesture will be recognized.	
	 @return PXC_STATUS_NO_ERROR if the mapping was successful, an error otherwise.
	 */
	virtual pxcStatus PXCAPI AddGestureActionMapping(pxcCHAR* gestureName,Action action,ActionHandler* actionHandler) = 0;
	__inline pxcStatus AddGestureActionMapping(pxcCHAR* gestureName,Action action) {
		return AddGestureActionMapping(gestureName, action, 0);
	}
	
	/**
		@brief Clear all previous Gesture to Action mappings
		@return PXC_STATUS_NO_ERROR if the mapping was successful, an error otherwise.
	*/
	virtual pxcStatus PXCAPI ClearAllGestureActionMappings(void) = 0;
};
