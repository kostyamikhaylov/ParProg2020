#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>


double KahanSumReversed (double *arr, int len)
{
	double sum = 0.0, error = 0.0, y = 0.0, t = 0.0;
	for (int i = len - 1; i >= 0; i--)
	{
		y = arr[i] - error;
		t = sum + y;
		error = (t - sum) - y;
		sum = t;
	}
	return sum;
}

double calc (uint32_t x_last, uint32_t num_threads)
{
	double result = 0.0;
	double *res_arr = NULL, *fact_arr = 0;
	res_arr = (double *) calloc (x_last, sizeof (*res_arr));
	fact_arr = (double *) calloc (x_last, sizeof (*res_arr));
	if (!res_arr || !fact_arr)
	{
		fprintf (stderr, "Calloc error\n");
		exit (-1);
	}

    fact_arr[0] = 1;
    for (unsigned int i = 1; i < x_last; i++)
        fact_arr[i] = fact_arr[i - 1] * i;

	#pragma omp parallel for num_threads(num_threads)
		for (unsigned int i = 0; i < x_last; i++)
			res_arr[i] = 1.0 / fact_arr[i];

	result = KahanSumReversed (res_arr, x_last);
	free (res_arr);
	free (fact_arr);
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
