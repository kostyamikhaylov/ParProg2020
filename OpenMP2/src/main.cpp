#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>

double calc(uint32_t x_last, uint32_t num_threads)
{
	double result = 0.0;
	double *thr_res = NULL;
	thr_res = (double *) calloc (x_last, sizeof (double));
	if (!thr_res)
		exit (-1);

	#pragma omp parallel for num_threads (num_threads)
		for (int i = x_last; i >= 1; i--)
		{
			thr_res[i - 1] += 1.0 / i;
		}

	for (int i = x_last; i > 0; i--)
		result += thr_res[i - 1];

	return result;
}

int main(int argc, char** argv)
{
  // Check arguments
  if (argc != 3)
  {
    std::cout << "[Error] Usage <inputfile> <output file>\n";
    return 1;
  }

  // Prepare input file
  std::ifstream input(argv[1]);
  if (!input.is_open())
  {
    std::cout << "[Error] Can't open " << argv[1] << " for write\n";
    return 1;
  }

  // Prepare output file
  std::ofstream output(argv[2]);
  if (!output.is_open())
  {
    std::cout << "[Error] Can't open " << argv[2] << " for read\n";
    input.close();
    return 1;
  }

  // Read arguments from input
  uint32_t x_last = 0, num_threads = 0;
  input >> x_last >> num_threads;
  // Calculation
  double res = calc(x_last, num_threads);

  // Write result
  output << std::setprecision(15) << res << std::endl;
  // Prepare to exit
  output.close();
  input.close();
  return 0;
}
