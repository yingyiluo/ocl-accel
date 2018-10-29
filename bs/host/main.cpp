#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
using namespace aocl_utils;

// CL binary name
const char *binary_prefix = "bs";
// The set of simultaneous kernels
enum KERNELS {
  K_WRITER,
  K_SEARCH,
  K_NUM_KERNELS
};
static const char *kernel_names[K_NUM_KERNELS] =
{
  "writer",
  "bsearch"
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
void bs(long N);
long search_data(double data, long N);
ulong xorshift128plus(cl_ulong2 seed);
bool init();
void cleanup();

// Host memory buffers
double *h_inData;
ulong *h_outData;
// Device memory buffers
cl_mem d_inData, d_outData;

int main(int argc, char **argv) {
  long N = (1 << 28); //16M
  Options options(argc, argv);

  if(options.has("n")) {
    N = options.get<long>("n");
  }
  printf("Number of elements in the array is set to %ld\n", N);

  if (!init())
    return false;
  printf("Init complete!\n");

  // Allocate host memory
  h_inData = (double *)alignedMalloc(sizeof(double) * N);
  h_outData = (ulong *)alignedMalloc(sizeof(ulong) * 1);
  if (!(h_inData && h_outData)) {
    printf("ERROR: Couldn't create host buffers\n");
    return false;
  }

  // Test
  bs(N);

  cleanup();
  return 0;
}

// Test channel
void bs(long N) {
  // Initialize input and produce verification data
  for (long i = 0; i < N; i++) {
    h_inData[i] = (double)i;
  }

  // Create device buffers - assign the buffers in different banks for more efficient
  // memory access 
  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * N, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");
  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_ulong), NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  // Copy data from host to device
  status = clEnqueueWriteBuffer(queues[K_SEARCH], d_inData, CL_TRUE, 0, sizeof(double) * N, h_inData, 0, NULL, NULL);
  checkError(status, "Failed to copy data to device");

  // Set the kernel arguments
  // Writer
  cl_ulong2 seed;
  seed.x = 5L;
  seed.y = 6L;
  status = clSetKernelArg(kernels[K_WRITER], 0, sizeof(cl_ulong2), (void*)&seed);
  checkError(status, "Failed to set kernel_writer arg 0");
  status = clSetKernelArg(kernels[K_WRITER], 1, sizeof(cl_long), (void*)&N);
  checkError(status, "Failed to set kernel_writer arg 1");
  // Search
  status = clSetKernelArg(kernels[K_SEARCH], 0, sizeof(cl_mem), (void*)&d_inData);
  checkError(status, "Failed to set kernel_wr arg 0");
  status = clSetKernelArg(kernels[K_SEARCH], 1, sizeof(cl_mem), (void*)&d_outData);
  checkError(status, "Failed to set kernel_wr arg 1");

  double time = getCurrentTimestamp();
  cl_event kernel_event;
  //TODO: compare with clEnqueueNDRangeKernel when using channel
  // Write
  status = clEnqueueTask(queues[K_WRITER], kernels[K_WRITER], 0, NULL, NULL);
  checkError(status, "Failed to launch kernel_writer");
  // Search
  status = clEnqueueTask(queues[K_SEARCH], kernels[K_SEARCH], 0, NULL, &kernel_event);
  checkError(status, "Failed to launch kernel_search");
  clWaitForEvents(1, &kernel_event);
  for(int i=0; i<K_NUM_KERNELS; ++i) {
    status = clFinish(queues[i]);
    checkError(status, "Failed to finish (%d: %s)", i, kernel_names[i]);
  }

  // Record execution time
  time = getCurrentTimestamp() - time;

  // Copy results from device to host
  status = clEnqueueReadBuffer(queues[K_SEARCH], d_outData, CL_TRUE, 0, sizeof(ulong), h_outData, 0, NULL, NULL);
  checkError(status, "Failed to copy data from device");

  printf("\nVerifying\n");
  int M = 100;
  ulong cs = 0;
  for(int i = 0; i < M; i++) {
    double target = (double) (xorshift128plus(seed) % N);
    long pos = search_data(target, N);
    cs ^= pos;
  }
  printf("cs: %ld, h_ouData: %ld\n", cs, *h_outData);
  if(cs != *h_outData) 
    printf("Verification Failed\n");
  else
    printf("Verification Succeeded\n");
  printf("\nProcessing time = %.4fms\n", (float)(time * 1E3));
}

long search_data(double data, long length)
{
        long low = 0;
        long high = length - 1;
        long mid = (low + high)/2;

   while(low <= high){
      if(h_inData[mid] < data){
         low = mid + 1;
      }
      else if(h_inData[mid] == data){
         return mid;
      }
      else if(h_inData[mid] > data){
         high = mid - 1;
      }
      mid = (low + high)/2;
   }

   return mid;
}

ulong xorshift128plus(cl_ulong2 s)
{
        ulong x = s.x;
        ulong const y = s.y;

        s.x = y;
        x ^= x << 23; // a
        s.y = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c

        return s.y + y;
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
