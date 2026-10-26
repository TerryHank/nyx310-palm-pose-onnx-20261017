#pragma once
#include <functional>
#include <cstdint>

#ifdef ALG_EXPORT_ON
#ifdef _WIN32
#define ALG_API_EXPORT __declspec(dllexport)
#else
#define ALG_API_EXPORT __attribute__((visibility("default")))
#endif
#else
#ifdef _WIN32
#define ALG_API_EXPORT __declspec(dllimport)
#else
#define ALG_API_EXPORT __attribute__((visibility("default")))
#endif
#endif

#ifdef __cplusplus
#define ALG_C_API_EXPORT extern "C" ALG_API_EXPORT
#else
#define ALG_C_API_EXPORT ALG_API_EXPORT
#endif

//Specifies the type of image frame.
typedef enum FrameTypeEnum
{
    FRAME_DEPTH = 0,                                    //Depth frame with 16 bits per pixel in millimeters.
    FRAME_IR = 1,                                       //IR frame with 8 bits per pixel.
    FRAME_RGB = 2,                                      //Color frame with 24 bits per pixel in RGB format.
    FRAME_TRANSFORM_DEPTH_IMG_TO_COLOR_SENSOR = 3,      //Depth frame with 16 bits per pixel, in millimeters, that is transformed to color sensor
                                                        //space where the resolution is same as the color frame's resolution.
    FRAME_TRANSFORM_COLOR_IMG_TO_DEPTH_SENSOR = 4,      //Color frame with 24 bits per pixel in RGB format, that is transformed to depth
                                                        //sensor space where the resolution is the same as the depth frame's resolution.

}FRAME_TYPE_E;

//Specifies the image pixel format.
typedef enum PixelFormatEnum
{
	PIXEL_FORMAT_DEPTH_MM16 = 0,      //Depth image pixel format, 16 bits per pixel in mm.
    PIXEL_FORMAT_GRAY_8 = 2,          //Gray image pixel format, 8 bits per pixel
    PIXEL_FORMAT_RGB_888 = 5,         //Without compress, color image pixel format, 24 bits per pixel RGB format.    
}PIXEL_FORMAT_E;

// Return status codes for all APIs.
// ALGO_RET_OK means the API successfully completed its operation.
// All other codes indicate a device, parameter, or API usage error.
typedef enum AlgorithmReturnEnum
{
    ALGO_RET_OK = 0,                                //The function completed successfully.
    ALGO_RET_NG = -1,                               //The function failure.
    ALGO_RET_LOAD_CONFIG_FILE_NG = -2,              //The configuration file failed to load.
    ALGO_RET_CHECK_PARAMS_NG = -3,                  //Parameters checking failed.
    ALGO_RET_INIT_MODEL_NG = -4,                    //Model initialization failed.
    ALGO_RET_GET_FRAME_TIMEOUT = -5,                //Frame acquisition timeout.
    ALGO_RET_CHECK_FRAME_NG = -6,                   //Frame check failed.
    ALGO_RET_NOT_SUPPORT = -7,                      //Operation not supported.
    ALGO_RET_NOT_INIT = -8,                         //The library is uninitialized.
    ALGO_RET_JSON_PARSE_NG = -9,                    //An error occurred while parsing json content.
    ALGO_RET_JSON_GENERATE_NG = -10,                //Failed to generate json format string.
    ALGO_RET_CHECK_FRAME_COUNT_NG = -11,            //Frame count error
    ALGO_RET_CHECK_FRAME_INDEX_NG = -12,            //Frame index error
    ALGO_RET_CHECK_FRAME_DEPTH_PARAMS_NG = -13,     //The parameters of the depth frame are wrong
    ALGO_RET_CHECK_FRAME_IR_PARAMS_NG = -14,        //The parameters of the ir frame are wrong
    ALGO_RET_CHECK_FRAME_RGB_PARAMS_NG = -15,       //The parameters of the rgb frame are wrong
    ALGO_RET_CHECK_FRAME_SUPPORT_NG = -16,          //The type of frame is not supported

    //TODO: more err_code ...
}ALGO_RET_E;

// Specifies the running state of the algorithm
typedef enum AlgorithmStateEnum
{
    ALGO_STATE_STOP = 0,                    //The algorithm has stopped.
    ALGO_STATE_RUNNING = 1,                 //The algorithm is running.
    ALGO_STATE_GET_FRAME_TIMEOUT = 5,       //The algorithm gets a frame timeout
    //TODO: more state ...
}ALGO_STATE_E;

//Specifies the ID of the parameter
typedef enum ParamID
{
    PARAM_VERSION = 0x00,                      // The version of the Algorithm library
    PARAM_NAME = 0x01,                         // The name of the Algorithm library
    
    PARAM_SAVE_RESULT_IMAGE_ENABLE = 0x10,     //Save an image of the model predictions
    PARAM_TEST_ASCII = 0x11,                   //Test ASCII parameter
    PARAM_TEST_HEX = 0x12,                     //Test HEX parameter

    PARAM_CONF_THRESH = 0x21,                  //The threshold of the prediction box
    PARAM_IOU_THRESH = 0x22,                   //The threshold for IOU detection
}PARAM_ID_E;

typedef enum OutputType
{
    OUTPUT_NONE = 0,
    OUTPUT_INTERNET = 0x01,
    OUTPUT_GPIO = 0x02,
    OUTPUT_RS485 = 0x04,
    OUTPUT_LED = 0x08,
    OUTPUT_ALL = OUTPUT_INTERNET | OUTPUT_GPIO | OUTPUT_RS485 | OUTPUT_LED,
}OUPUT_TYPE_E;

typedef enum OutputHandleType
{
    OUTPUT_HANDLE_LED = 0,
    OUTPUT_HANDLE_GPIO = 1,
    OUTPUT_HANDLE_UART = 2,
} OUTPUT_HANDLE_TYPE_E;

typedef enum Baudrate{
    B0 = 0,       //hang up
    B50 = 1,
    B75  = 2,
    B110 = 3,
    B134 = 4,
    B150 = 5,
    B200 = 6,
    B300 = 7,
    B600 = 8,
    B1200 = 9,
    B1800 = 10,
    B2400 = 11,
    B4800 = 12,
    B9600 = 13,
    B19200 = 14,
    B38400 = 15,
    B57600 = 16,
    B115200 = 17,
    B230400 = 18,
    B460800 = 19,
    B500000 = 20,
    B576000 = 21,
    B921600 = 22,
    B1000000 = 23,
    B1152000 = 24,
    B1500000 = 25,
    B2000000 = 26,
    B2500000 = 27,
    B3000000 = 28,
    B3500000 = 29,
    B4000000 = 30
}BAUDRATE_E;


#pragma pack (push, 1)

//image frame data
typedef struct CameraSingleFrame
{
    uint32_t frame_no;
    uint64_t timestamp;
    FRAME_TYPE_E frame_type;
    PIXEL_FORMAT_E pixel_format;
    uint32_t width;
    uint32_t height;
    uint32_t data_len;
    uint8_t *p_data;
}CAM_SINGLE_FRAME_T;

//intrinsic parameters and external parameters for tof lens and color lens 
typedef struct LensParameters
{
    float tofIntrinsic[4];     //intrinsic parameters for tof lens: cx,cy,fx,fy
    float colorIntrinsic[4];   //intrinsic parameters for color lens:cx,cy,fx,fy
	float rotation[9];         //Orientation stored as an array of 9 float representing a 3x3 rotation matrix. The rotation matrix from ToF lens to color lens.
    float translation[3];      //Location stored as an array of 3 float representing a 3-D translation vector. The translation matrix from ToF lens to color lens.
}LENS_PARAMETERS_T;

//image frame data and lens's parameters
//When multiple frames of synchronized images are transmitted, the images are stored in image type (@FrameTypeEnum) order.
//For example, when both the color image and the transformed depth image are transferred,
//p_frame[0] holds the color image and p_frame[1] holds the transformed depth image. 
typedef struct CameraJointFrame
{
    uint8_t frame_cnt;                  //the count of frames
    CAM_SINGLE_FRAME_T *p_frame;        //frame data array
    LENS_PARAMETERS_T lens_parameters;  //the Parameters of Lens
}CAM_JOINT_FRAME_T;

//Initialization parameters of the algorithm
typedef struct AlgorithmInitParam
{
    /*The root directory of the algorithm module. The default root is /userdata/algorithm_data.
    A typical directory structure is as follows:
    
    ./
    |-- lib
    |   `-- libalg.so
    `-- model
        `-- yolov5s_relu_precompile.rknn
    */
    char *p_alg_root_dir;

    using ALGO_PROCESS_CB_FUNC_T = std::function<void (uint64_t timestamp, const char *p_proc_result, uint32_t result_len)>; 
	ALGO_PROCESS_CB_FUNC_T process_cb_func;
}ALGO_INIT_PARAM_T;

#pragma pack (pop)