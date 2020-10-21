#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>

double calc (uint32_t x_last, uint32_t num_threads)
{
	double result = 0.0;
	double *res_arr = NULL, *exchange = NULL;
	res_arr = (double *) calloc (x_last, sizeof (*res_arr));
	exchange = (double *) calloc (num_threads, sizeof (*exchange));
	if (!res_arr || !exchange)
	{
		fprintf (stderr, "Calloc error\n");
		exit (-1);
	}

	#pragma omp parallel num_threads (num_threads)
	{
		double fact = 1.0;
		int num = omp_get_thread_num ();
		unsigned int i = 0;
		#pragma omp for
		for (i = 1; i < x_last; i++)
		{
			fact *= i;
			res_arr[i - 1] = fact;
		}

		exchange[num] = fact;

		#pragma omp barrier

		fact = 1.0;
		for (; num > 0; num--)
			fact *= exchange[num - 1];

		#pragma omp for
		for (i = 0; i < x_last - 1; i++)
		{
			res_arr[i] *= fact;
		}
		
		#pragma omp barrier

		#pragma omp for
		for (i = 0; i < x_last - 1; i++)
			res_arr[i] = 1.0 / res_arr[i];
	}

	for (int i = x_last - 2; i >= 0; i--)
		result += res_arr[i];
	result += 1.0;

	free (res_arr);
	free (exchange);
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
  output << std::setprecision(16) << res << std::endl;
  // Prepare to exit
  output.close();
  input.close();
  return 0;
}
