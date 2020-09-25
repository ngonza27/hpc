#include <mkl.h>
#include <hbwmalloc.h>

//implement scratch buffer on HBM and compute FFTs, refer instructions on Lab page

// MKL_Complex8 *data = (MKL_Complex8 *) _mm_malloc(sizeof(MKL_Complex8)*num_fft*fft_size, 4096);
// MKL_Complex8 *ref_data = (MKL_Complex8 *)_mm_malloc(sizeof(MKL_Complex8) * num_fft * fft_size, 4096);
// ref_data[i].real = data[i].real;
// ref_data[i].imag = data[i].imag;

void runFFTs(const size_t fft_size,
             const size_t num_fft,
             MKL_Complex8 *data,
             DFTI_DESCRIPTOR_HANDLE *fftHandle)
{
  const long buff_size = 1 << 27;
  MKL_Complex8 *buff;
  hbw_posix_memalign((void **)&buff, 4096, sizeof(MKL_Complex8) * buff_size);

  for (size_t i = 0; i < num_fft; ++i)
  {
#pragma omp parallel for
    for (size_t j = 0; j < fft_size; ++j)
    {
      buff[j].real = data[i * fft_size + j].real;
      buff[j].imag = data[i * fft_size + j].imag;
    }
    DftiComputeForward(*fftHandle, buff);
#pragma omp parallel for
    for (size_t j = 0; j < fft_size; ++j)
    {
      data[i * fft_size + j].real = buff[j].real;
      data[i * fft_size + j].imag = buff[j].imag;
    }
  }
  hbw_free(buff);
}