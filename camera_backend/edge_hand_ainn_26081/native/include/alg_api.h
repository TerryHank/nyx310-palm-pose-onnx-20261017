#pragma once

#include "alg_define.h"

/**
 * @brief        Initialize the algorithm library. This function must be invoked before any other APIs.
 * @param[in]    p_init_param     The pointer of the initialization parameter
 * @return       ::ALGO_RET_OK    If the function succeeded, or one of the error values defined by ::ALGO_RET_E.
 */
ALG_C_API_EXPORT ALGO_RET_E algo_initialize(const ALGO_INIT_PARAM_T *p_init_param);

/**
 * @brief        Start the processing thread of the algorithm
 * @return       ::ALGO_RET_OK    If the function succeeded, or one of the error values defined by ::ALGO_RET_E.
 */
ALG_C_API_EXPORT ALGO_RET_E algo_ctrl_start(void);

/**
 * @brief        Stop the processing thread of the algorithm
 * @return       ::ALGO_RET_OK    If the function succeeded, or one of the error values defined by ::ALGO_RET_E.
 */
ALG_C_API_EXPORT ALGO_RET_E algo_ctrl_stop(void);

/**
 * @brief        Push frame data to the algorithm
 * @param[in]    p_joint_frame    The pointer of the input frame
 * @return       ::ALGO_RET_OK    If the function succeeded, or one of the error values defined by ::ALGO_RET_E.
 */
ALG_C_API_EXPORT ALGO_RET_E algo_push_frame(const CAM_JOINT_FRAME_T *p_joint_frame);

/**
 * @brief        Set the parameters of the algorithm
 * @param[in]    p_in_buf         The pointer to the input parameters' buffer.
 * @param[in]    in_buf_len       The size, in bytes, of the parameter value contained in p_in_buf.
 * @param[out]   p_out_buf        The pointer to the output buffer.
 * @param[out]   p_out_buf_len    The pointer to the output buffer vaild size in bytes.
 * @return       ::ALGO_RET_OK    If the function succeeded, or one of the error values defined by ::ALGO_RET_E.
 */
ALG_C_API_EXPORT ALGO_RET_E algo_set_param(const char *p_in_buf, uint16_t in_buf_len, const char **p_out_buf, uint16_t *p_out_buf_len);

/**
 * @brief        Get the parameters of the algorithm
 * @param[in]    p_in_buf         The pointer to the input parameters' buffer.
 * @param[in]    in_buf_len       The size, in bytes, of the parameter value contained in p_in_buf.
 * @param[out]   p_out_buf        The pointer to the output buffer.
 * @param[out]   p_out_buf_len    The pointer to the output buffer vaild size in bytes.
 * @return       ::ALGO_RET_OK    If the function succeeded, or one of the error values defined by ::ALGO_RET_E.
 */
ALG_C_API_EXPORT ALGO_RET_E algo_get_param(const char *p_in_buf, uint16_t in_buf_len, const char **p_out_buf, uint16_t *p_out_buf_len);

/**
 * @brief        Sets the output handle for the specified output type
 * @param[in]    outputHandleType          The specified output type
 * @param[in]    p_output_handle  The output handle for the specified output type
 * @return       ::ALGO_RET_OK    If the function succeeded, or one of the error values defined by ::ALGO_RET_E.
 */
ALG_C_API_EXPORT ALGO_RET_E algo_set_output_handle(OutputHandleType outputHandleType, void *p_output_handle);

/**
 * @brief        Releases the resources used by the algorithm library. After invoking this function, no other APIs should be invoked except algo_initialize..
 * @return       ::ALGO_RET_OK    If the function succeeded, or one of the error values defined by ::ALGO_RET_E.
 */
ALG_C_API_EXPORT ALGO_RET_E algo_release(void);