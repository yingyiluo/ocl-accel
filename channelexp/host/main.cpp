#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
using namespace aocl_utils;

// CL binary name
const char *binary_prefix = "channeltest";
// The set of simultaneous kernels
enum KERNELS {
  K_WRITER,
  K_READER,
  K_NUM_KERNELS
};
static const char *kernel_names[K_NUM_KERNELS] =
{
  "data_in",
  "data_out"
};

// ACL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queues[K_NUM_KERNELS];
static cl_kernel kernels[K_NUM_KERNELS];
static cl_program program = NULL;
static cl_int status = 0;

// Function prototypes
void test();
bool init();
void cleanup();

// Host memory buffers
double *h_inData, *h_outData, *h_verify;
// Device memory buffers
cl_mem d_inData, d_outData;

int main(int argc, char **argv) {
  if (!init())
    return false;
  printf("Init complete!\n");

  // Allocate host memory
  int N = 50;
  h_inData = (double *)alignedMalloc(sizeof(double) * N);
  h_outData = (double *)alignedMalloc(sizeof(double) * N);
  h_verify = (double *)alignedMalloc(sizeof(double) * N);
  if (!(h_inData && h_outData && h_verify)) {
    printf("ERROR: Couldn't create host buffers\n");
    return false;
  }

  // Test
  test();

  cleanup();
  return 0;
}

// Test channel
void test() {
  int N = 50;
  // Initialize input and produce verification data
  for (int i = 0; i < N; i++) {
    h_inData[i] = (double)i;
  }

  // Create device buffers - assign the buffers in different banks for more efficient
  // memory access 
  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * N, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");
  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(double) * N, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  // Copy data from host to device
  status = clEnqueueWriteBuffer(queues[K_WRITER], d_inData, CL_TRUE, 0, sizeof(double) * N, h_inData, 0, NULL, NULL);
  checkError(status, "Failed to copy data to device");

  // Set the kernel arguments
  // Read
  status = clSetKernelArg(kernels[K_WRITER], 0, sizeof(cl_mem), (void*)&d_inData);
  checkError(status, "Failed to set kernel_rd arg 0");
  // Write
  status = clSetKernelArg(kernels[K_READER], 0, sizeof(cl_mem), (void*)&d_outData);
  checkError(status, "Failed to set kernel_wr arg 0");

  double time = getCurrentTimestamp();
  //cl_event kernel_event;
  //TODO: compare with clEnqueueNDRangeKernel when using channel
  // Write
  status = clEnqueueTask(queues[K_WRITER], kernels[K_WRITER], 0, NULL, NULL);
  checkError(status, "Failed to launch kernel_read");
  // Read
  status = clEnqueueTask(queues[K_READER], kernels[K_READER], 0, NULL, NULL);
  checkError(status, "Failed to launch kernel_write");
  //clWaitForEvents(1, &kernel_event);
  for(int i=0; i<K_NUM_KERNELS; ++i) {
    status = clFinish(queues[i]);
    checkError(status, "Failed to finish (%d: %s)", i, kernel_names[i]);
  }

  // Record execution time
  time = getCurrentTimestamp() - time;

  // Copy results from device to host
  status = clEnqueueReadBuffer(queues[0], d_outData, CL_TRUE, 0, sizeof(double) * N, h_outData, 0, NULL, NULL);
  checkError(status, "Failed to copy data from device");
  printf("\nhost\n");
  for(int i = 0; i < N; i++)
    printf("%f ", h_outData[i]);
  printf("\nProcessing time = %.4fms\n", (float)(time * 1E3));
}

// Set up the context, device, kernels, and buffers...
bool init() {
  cl_int status;

  // Start everything at NULL to help identify errors
  for(int i = 0; i < K_NUM_KERNELS; ++i){
    kernels[i] = NULL;
    queues[i] = NULL;
  }

  // Locate files via. relative paths
  if(!setCwdToExeDir())
    return false;

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform\n");
    return false;
  }

  // Query the available OpenCL devices and just use the first device if we find
  // more than one
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;
  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  device = devices[0];

  // Create the context.
  context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the command queues
  for(int i=0; i<K_NUM_KERNELS; ++i) {
    queues[i] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue (%d)", i);
  }

  // Create the program.
  std::string binary_file = getBoardBinaryFile(binary_prefix, device);
  printf("Using AOCX: %s\n\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  for(int i=0; i<K_NUM_KERNELS; ++i) {
    kernels[i] = clCreateKernel(program, kernel_names[i], &status);
    checkError(status, "Failed to create kernel (%d: %s)", i, kernel_names[i]);
  }

  return true;
}

// Free the resources allocated during initialization
void cleanup() {
  for(int i=0; i<K_NUM_KERNELS; ++i)
    if(kernels[i]) 
      clReleaseKernel(kernels[i]);  
  if(program) 
    clReleaseProgram(program);
  for(int i=0; i<K_NUM_KERNELS; ++i)
    if(queues[i]) 
      clReleaseCommandQueue(queues[i]);
  if(context) 
    clReleaseContext(context);
}
